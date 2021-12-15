"""Module containing all functions to featurize. 

Featurization functions

Featurizers should explicitly act on one hot encoded proteins and smiles
strings, which is how all data 

"""

import os
from typing import List, Optional, Tuple
import logging
from abc import ABC, abstractmethod
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import pickle

import torch
from bepler_embedding import embed_utils as be
from tape import ProteinBertModel, UniRepModel, TAPETokenizer
from tape import datasets as tape_dataset

from enzpred.features import alphabet
from enzpred.utils import file_utils, parse_utils, ssa_utils

FEATURIZER_ARGS = [
    (
        ["--n-bits"],
        dict(
            action="store",
            type=int,
            default=1024,
            help="Integer number of for fingerprint",
        ),
    ),
    (
        ["--ngram-min"],
        dict(
            action="store", type=int, default=2, help="Maximum number of ngrams to use"
        ),
    ),
    (
        ["--ngram-max"],
        dict(action="store", type=int, default=3, help="Minimum size of ngrams to use"),
    ),
    (
        ["--unnormalized"],
        dict(
            action="store_true", default=False, help="If true, use unnormalized kmers"
        ),
    ),
    (
        ["--pool-prot-strategy"],
        dict(
            action="store",
            type=str,
            default=None,
            help="""If true, pool the output of the protein embedding to have
          non-position specific embeddings""",
            choices=[
                None,
                "mean",
                "ssa",
                "rand",
                "hard",
                "randmsa",
                "msacover",
                "msaconserv",
                "hardcat",
                "randhard",
                "attn",
                "contact",
            ],
        ),
    ),
    (
        ["--pool-num"],
        dict(
            action="store",
            type=int,
            default=5,
            help="""If this value is set, pool pool_num residue embeddings""",
        ),
    ),
    (
        ["--embed-batch-size"],
        dict(action="store", type=int, default=4, help="""Size of batch embedding"""),
    ),
    (
        ["--cache-dir"],
        dict(
            action="store",
            type=str,
            default=None,
            help="""Directory to hold cached feature files""",
        ),
    ),
    (
        ["--chem-fp-file"],
        dict(
            action="store",
            default=None,
            type=str,
            help="""Name of pickled chem fingerprints""",
        ),
    ),
    (
        ["--prot-feat-file"],
        dict(
            action="store",
            default=None,
            type=str,
            help="""Name of pickled chem fingerprints""",
        ),
    ),
    (
        ["--evotuned-dir"],
        dict(
            action="store",
            default=None,
            type=str,
            help="""Directory containing evotuned tape model""",
        ),
    ),
    (
        ["--n-bits-prot"],
        dict(
            action="store",
            default=100,
            type=int,
            help="""Number of bits to use for random protein embeddings as control""",
        ),
    ),
    (
        ["--seq-msa"],
        dict(
            action="store",
            default=None,
            type=str,
            help="""Mltiple sequence alignment for featurizer""",
        ),
    ),
    (
        ["--jt-vae-loc"],
        dict(
            action="store",
            default="data/processed/precomputed_features/",
            type=str,
            help="""Location of the JT-VAE embeddings""",
        ),
    ),
]

CHEM_FEATURIZER_TYPES = [
    "smiles",
    "precomputed",
    "random",
    "jtvae",
    "maccs",
    "morgan1024",
    "morgan2048",
    "cat",
    None,
]

PROT_FEATURIZER_TYPES = [
    "kmer",
    "onehot",
    "bert",
    "unirep",
    "bepler",
    "precomputed",
    "random",
    "esm",
    "cat",
    "msa",
    "gapped",
    "str",
    "msafull",
    None,
]

# Helper fn
def none_fn(*args, **kwargs):
    None


def get_prot_featurizer(prot_featurizer: str, **kwargs):
    """Get protein featurers"""
    return {
        "kmer": KMERFeaturizer,
        "onehot": Uniprot21Featurizer,
        "msa": Uniprot21FeaturizerMSA,
        "gapped": MSAFeaturizer,
        "cat": CategoricalProt,
        "bert": BERTFeaturizer,
        "bepler": BeplerFeaturizer,
        "unirep": UniRepFeaturizer,
        "esm": ESMFeaturizer,
        "precomputed": PrecomputedProt,
        "random": RandomProt,
        "str": StrFeaturizer,
        "msafull": FullMSAFeaturizer,
        None: none_fn,
    }[prot_featurizer](**kwargs)


def get_chem_featurizer(chem_featurizer: str, **kwargs):
    """Get chem features"""
    return {
        "maccs": MACCSFeaturizer,
        "morgan1024": Morgan1024Featurizer,
        "morgan2048": Morgan2048Featurizer,
        "cat": CategoricalChem,
        "smiles": ChemIdentity,
        "precomputed": PrecomputedChem,
        "jtvae": JTVAEFeaturizer,
        "random": RandomChem,
        None: none_fn,
    }[chem_featurizer](**kwargs)


#### Generic Featurizers
class GenericFeaturizer(ABC):
    """GenericFeaturizer.
    use this to build different types of featurizers

    """

    def __init__(self, cache_dir: Optional[str] = None, **kwargs):
        """__init__.

        Args:
            cache_dir (str): Directory to cache featurizations
            kwargs: **kwargs

        """
        self.cache_dir = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_dir = cache_dir
        self.feature_cache = {}
        # Variable to have a set outdim
        self.out_dim_set = None
        self.is_fit = False
        super().__init__()

    @abstractmethod
    def featurize(self, obj_list: List[str]) -> List[np.ndarray]:
        """featurize.

        Args:
            obj_list (List[str]): obj_list containing objs to featurize

        Returns:
            np.ndarray: features
        """
        raise NotImplementedError()

    @abstractmethod
    def out_dim(self) -> Tuple[Optional[int]]:
        """out_dim.

        Get the out dimension of this featurizer.
        If it's variable, return None

        Returns:
            Tuple[Optional[int]]
        """
        raise NotImplementedError()

    def set_out_dim(self, val: Optional[int]):
        """set_out_dim.

        Args:
            val (Optional[int]): val
        """
        self.out_dim_set = (val,)

    def _fill_feature_cache(self, model_cache_str, feature_cache_dict):
        """Internal method"""
        if self.cache_dir:
            if model_cache_str is None:
                logging.info("Unable to laod feature cache due to empty cache_str")
                return

            cache_file = os.path.join(self.cache_dir, model_cache_str)
            if os.path.isfile(cache_file):
                logging.info(f"Loading cache file {cache_file}")
                try:
                    my_obj = file_utils.pickle_load(cache_file)
                except pickle.UnpicklingError:
                    logging.info(f"Unable to load {cache_file} due to pickling error")
                    my_obj = {}
                except EOFError:
                    logging.info(f"Unable to load {cache_file} due to EOF error")
                    my_obj = {}
                if isinstance(my_obj, dict):
                    feature_cache_dict.update(my_obj)
                    logging.info(
                        f"Loaded feature cache of size {len(feature_cache_dict)}"
                    )
                    return

        logging.info("No cache features loaded")

    def _dump_features(self, model_cache_str, feature_cache_dict):
        if self.cache_dir and model_cache_str:
            cache_file = os.path.join(self.cache_dir, model_cache_str)
            file_utils.pickle_obj(feature_cache_dict, cache_file)

    def fill_feature_cache(self):
        """Update the feature cache"""

        model_cache_str = self.get_model_props()
        if self.cache_dir and not self.get_model_props():
            logging.info(
                "No method get_model_props() implemented for featurizer {str(self)}"
            )
            return
        self._fill_feature_cache(model_cache_str, self.feature_cache)

    def dump_features(self):
        """Update the feature cache"""
        model_cache_str = self.get_model_props()
        self._dump_features(model_cache_str, self.feature_cache)

    def get_model_props(self) -> str:
        """Use this to encode the model into a cache file unique key"""
        raise NotImplementedError()

    def set_featurize(self, obj_list: List[str]) -> np.ndarray:
        """set_featurize.

        Convert obj_list into featurized array but don't repeat redundant
        calculations

        Args:
            obj_list (List[str]): obj_list

        Returns:
            np.ndarray
        """
        unique_inputs = list(set(obj_list))
        # Mapping from input item to its representation
        featurized_outputs = self.featurize(unique_inputs)
        if type(featurized_outputs) is not list:
            featurized_outputs = featurized_outputs.tolist()
        item_map = dict(zip(unique_inputs, featurized_outputs))
        return_list = []
        for item in obj_list:
            return_list.append(item_map[item])
        return np.array(return_list)


class PrecomputedFeaturizer(GenericFeaturizer):
    """Precomputed featurizer.
    Use this to load in features from a baseline
    """

    def __init__(self, pickle_file: str, **kwargs):
        """__init__.

        Args:
            pickle_file (str): Location of pickled features
            kwargs: **kwargs

        """
        super().__init__(**kwargs)
        self.in_file = pickle_file
        self.feature_map = file_utils.pickle_load(pickle_file)
        self.feat_bits = len(next(iter(self.feature_map.values())))

    def featurize(self, obj_list: List[str]) -> List[np.ndarray]:
        """featurize.

        Args:
            obj_list (List[str]): obj_list containing objs to featurize

        Returns:
            np.ndarray: features
        """
        my_l = []
        for j in obj_list:
            feats = self.feature_map.get(j, None)
            my_l.append(np.array(feats))
            if feats is None:
                raise ValueError(f"Cannot find features for {j}")
        return my_l

    def out_dim(self) -> Tuple[Optional[int]]:
        """out_dim.

        Get the out dimension of this featurizer.
        If it's variable, return None

        Returns:
            Tuple[Optional[int]]
        """
        if self.out_dim_set is not None:
            return self.out_dim_set
        else:
            return (self.feat_bits,)

    def fill_feature_cache(self):
        """Update the feature cache"""
        pass

    def dump_features(self):
        """Update the feature cache"""
        pass

    def get_model_props(self) -> str:
        """Use this to encode the model into a cache file unique key"""
        return {"cache_name": self.in_file}


class RandomFeaturizer(GenericFeaturizer):
    """Precomputed featurizer.
    Use this to load in features from a baseline
    """

    def __init__(self, random_size: int, **kwargs):
        """__init__.

        Args:
            random_size (int): Size of random fingerprints for each item
            kwargs: **kwargs

        """
        super().__init__(**kwargs)
        self.n_bits = random_size
        self.feature_map = {}

    def featurize(self, obj_list: List[str]) -> List[np.ndarray]:
        """featurize.

        Args:
            obj_list (List[str]): obj_list containing objs to featurize

        Returns:
            np.ndarray: features
        """
        my_l = []
        for j in obj_list:
            if j in self.feature_map:
                my_l.append(self.feature_map[j])
            else:
                self.feature_map[j] = np.random.random(self.n_bits)
                my_l.append(self.feature_map[j])

        return np.array(my_l)

    def out_dim(self) -> Tuple[Optional[int]]:
        """out_dim.

        Get the out dimension of this featurizer.
        If it's variable, return None

        Returns:
            Tuple[Optional[int]]
        """
        if self.out_dim_set is not None:
            return self.out_dim_set
        else:
            return (self.n_bits,)

    def fill_feature_cache(self):
        """Update the feature cache"""
        pass

    def dump_features(self):
        """Update the feature cache"""
        pass

    def get_model_props(self) -> str:
        """Use this to encode the model into a cache file unique key"""
        return {}


class CategoricalFeaturizer(GenericFeaturizer):
    """Categorial featurizer."""

    def __init__(self, max_examples: int = 500, **kwargs):
        """__init__.

        Args:
            max_examples: Size of bit vector
            kwargs: **kwargs

        """
        super().__init__(**kwargs)
        self.max_examples = max_examples
        self.current_index = 0
        self.n_bits = self.max_examples
        self.feature_map = {}

    def featurize(self, obj_list: List[str]) -> List[np.ndarray]:
        """featurize.

        Args:
            obj_list (List[str]): obj_list containing objs to featurize

        Returns:
            np.ndarray: features
        """
        my_l = []
        for j in obj_list:
            if j in self.feature_map:
                my_l.append(self.feature_map[j])
            else:

                vec = np.zeros(self.n_bits)
                vec[self.current_index] = 1
                self.current_index += 1

                if self.current_index >= self.max_examples:
                    raise ValueError(
                        "Trying to cateogrically featurize more examples than intended. Raise max_examples"
                    )

                self.feature_map[j] = vec

                my_l.append(self.feature_map[j])

        return np.array(my_l)

    def out_dim(self) -> Tuple[Optional[int]]:
        """out_dim.

        Get the out dimension of this featurizer.
        If it's variable, return None

        Returns:
            Tuple[Optional[int]]
        """
        if self.out_dim_set is not None:
            return self.out_dim_set
        else:
            return (self.max_examples,)

    def fill_feature_cache(self):
        """Update the feature cache"""
        pass

    def dump_features(self):
        """Update the feature cache"""
        pass

    def get_model_props(self) -> str:
        """Use this to encode the model into a cache file unique key"""
        return {}


#### Chemical featurizers


class ChemFeaturizer(GenericFeaturizer):
    """ChemFeaturizer.
    Abstract class

    Use this to build different chem featurizers
    """

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """
        super().__init__(**kwargs)


class RandomChem(RandomFeaturizer):
    """Base fingerprint featurzer
    Use this to abstract a random fingerprinter
    """

    def __init__(self, n_bits: int = 100, **kwargs):
        """__init__.

        Args:
            n_bits (int) : Number of bits
            kwargs: kwargs
        """

        super().__init__(n_bits, **kwargs)


class CategoricalChem(CategoricalFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CategoricalProt(CategoricalFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class FingerprintBaseFeaturizer(ChemFeaturizer):
    """Base fingerprint featurzer
    Use this to abstract RDKIT and map4 to one base class
    """

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """

        super().__init__(**kwargs)

    def out_dim(self) -> Tuple[Optional[int]]:
        """out_dim.

        Get the out dimension of this featurizer.
        If it's variable, return None

        Returns:
           Tuple[Optional[int]]
        """
        if self.out_dim_set is not None:
            # If set by a featurizer..
            return self.out_dim_set
        else:
            return (self.get_nbits(),)

    def get_nbits(self) -> int:
        """Implement number of bits"""
        raise NotImplementedError()

    def mol_to_fp(self, mol: Chem.Mol) -> np.ndarray:
        """Convert mol to fingerprint"""
        raise NotImplementedError()

    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        """featurize.

        Args:
            smiles_list (List[str]): smiles_list containing strings of smiles

        Returns:
            np.ndarray of features
        """

        self.fill_feature_cache()
        to_featurize = []
        smiles_mapping = {}
        for smiles in smiles_list:
            if smiles in self.feature_cache:
                smiles_mapping[smiles] = self.feature_cache[smiles]
            else:
                to_featurize.append(smiles)

        for mol_str in to_featurize:
            mol = Chem.MolFromSmiles(mol_str)
            array = self.mol_to_fp(mol)
            smiles_mapping[mol_str] = array

        ret_ar = []
        for mol in smiles_list:
            ret_ar.append(smiles_mapping[mol])

        # Dump features learned and update cache
        self.feature_cache.update(smiles_mapping)
        self.dump_features()

        return np.array(ret_ar)


class PrecomputedChem(PrecomputedFeaturizer):
    """Precomputed featurizer.
    Use this to load in features from a baseline
    """

    def __init__(self, chem_fp_file: str, **kwargs):
        """__init__.

        Args:
            chem_fp_file(str): Name of file that has fingerprints saved
                in pickle
            kwargs: kwargs
        """
        super().__init__(pickle_file=chem_fp_file, **kwargs)


class JTVAEFeaturizer(PrecomputedChem):
    """ChemFeaturizer.
    Abstract class

    Use this to build different chem featurizers
    """

    def __init__(self, hts_csv_file: str, jt_vae_loc: str, **kwargs):
        """__init__.

        Args:
            hts_csv_file (str): Name of csv file to extract dataset from
            jt_vae_loc (str): Location of jt vae pickled embeddings
            kwargs: kwargs
        """

        file_name = hts_csv_file.split("/")[-1]
        file_name = file_name.split(".")[0]

        # Get chunk used in jt vae loc
        prefix = file_name.split("_")[0]
        pickle_file = os.path.join(jt_vae_loc, f"{prefix}_JTVAE_features.p")

        kwargs["chem_fp_file"] = pickle_file

        super().__init__(**kwargs)


class RDFeaturizer(FingerprintBaseFeaturizer):
    """RDFeaturizer.
    Featurizer that returns any RDkit featur vector

    Use this to build different chem featurizers
    """

    def __init__(self, fprint_name: str = "MACCS", n_bits: int = 100, **kwargs):
        """__init__.

        Args:
            fprint_name (str): fprint_name
            n_bits (int): n_bits
            kwargs: kwargs
        """

        self.fprint_name = fprint_name

        if self.fprint_name == "MACCS":
            self.fp_fn = lambda m: AllChem.GetMACCSKeysFingerprint(m)
            self.nbits = 167
        elif self.fprint_name == "MORGAN":
            self.nbits = n_bits
            self.fp_fn = lambda m: AllChem.GetMorganFingerprintAsBitVect(
                m, 2, nBits=self.nbits, useChirality = True # Set 
            )
        else:
            raise Exception(f"Fingerprint {self.fprint_name} is not NotImplemented")
        super().__init__(**kwargs)

    def get_nbits(self) -> int:
        """Implement number of bits"""
        return self.nbits

    def mol_to_fp(self, mol: Chem.Mol) -> np.ndarray:
        """Convert mol to fingerprint"""
        fingerprint = self.fp_fn(mol)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

    def get_model_props(self):
        """Use this to encode the model into a cache file"""
        model_str = {"fprint_name": self.fprint_name, "nbits": self.nbits}
        hashed_str = file_utils.md5(str(model_str))
        return hashed_str


class MACCSFeaturizer(RDFeaturizer):
    """MACCSFeaturizer.

    Create MACCSFeaturizer that featurizes with maccs keys
    """

    def __init__(self, **kwargs):
        """__init__.
        Args:
            kwargs: kwargs
        """

        kwargs["n_bits"] = 167
        kwargs["fprint_name"] = "MACCS"
        super().__init__(**kwargs)


class Morgan1024Featurizer(RDFeaturizer):
    """Morgan1024Featurizer

    Morgan fingerprint of 1024 bits
    """

    def __init__(self, **kwargs):
        """__init__.
        Args:
            kwargs: kwargs
        """
        kwargs["n_bits"] = 1024
        kwargs["fprint_name"] = "MORGAN"
        super().__init__(**kwargs)


class Morgan2048Featurizer(RDFeaturizer):
    """Morgan2048Featurizer

    Morgan fingerprint of 2048 bits
    """

    def __init__(self, **kwargs):
        """__init__.
        Args:
            kwargs: kwargs
        """
        kwargs["n_bits"] = 2048
        kwargs["fprint_name"] = "MORGAN"
        super().__init__(**kwargs)

class ChemIdentity(ChemFeaturizer):
    """ChemIdentity.
    Featurizer that just returns smiles strings (identity mapping)

    """

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """

        super().__init__(**kwargs)

    def out_dim(self) -> Tuple[Optional[int]]:
        """out_dim.

        Get the out dimension of this featurizer.
        If it's variable, return None

        Returns:
           Tuple[Optional[int]]
        """
        return (None,)

    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        """featurize.

        Args:
            smiles_list (List[str]): smiles_list containing strings of smiles

        Returns:
            np.ndarray of features
        """
        return np.array(smiles_list)


###### Protein featurizers


class ProtFeaturizer(GenericFeaturizer):
    """ProtFeaturizer.
    Abstract class

    Use this to build different protein featurizers

    """

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """
        super().__init__(**kwargs)


class RandomProt(RandomFeaturizer):
    """Base fingerprint featurzer
    Use this to abstract a random fingerprinter
    """

    def __init__(self, n_bits_prot: int = 100, **kwargs):
        """__init__.

        Args:
            n_bits_prot (int) : Number of bits
            kwargs: kwargs
        """

        super().__init__(n_bits_prot, **kwargs)


class PrecomputedProt(PrecomputedFeaturizer):
    """Precomputed featurizer.
    Use this to load in features from a baseline
    """

    def __init__(self, prot_feat_file: str, **kwargs):
        """__init__.


        Args:
            prot_feat_file (str): Name of file that has fingerprints saved
                in pickle
            kwargs: kwargs
        """
        super().__init__(pickle_file=prot_feat_file, **kwargs)


class TAPEFeaturizer(ProtFeaturizer):
    """TAPEFeaturizer.

    Use TAPE to help featurize proteins.

    https://github.com/songlab-cal/tape

    """

    def __init__(
        self,
        pool_prot_strategy: Optional[str] = None,
        seq_msa: str = None,
        embed_batch_size: int = 8,
        gpu: bool = False,
        ssa_ref_file: str = None,
        pool_num: Optional[int] = None,
        **kwargs,
    ):
        """__init__.

        Args:
            pool_prot_strategy (str) : Strategy for how to pool output protein
                representations. If none, don't pool. Options: [None, "mean",
                "ssa", "hard", "hardcat", "randhard", "msarand", "attn",
                "contact", "rand"]
            seq_msa (str) : Multiple sequence alignment of all sequences to
                featurize
            embed_batch_size (int): Embedding size for TAPE. Default 8
            gpu (bool): If true, use gpu
            ssa_ref_file (str) : Name of file that contains both a reference
                protein and impt amino acids in that sequence
            pool_num (Optional[int]) : Number of amino acid residues to
                pool over; note we should still only select these positions
                from the reference sequence
            kwargs: kwargs
        """

        # If false, fit, otherwise just run
        super().__init__(**kwargs)

        self.embed_batch_size = embed_batch_size
        self.unpooled_feature_cache = {}
        self.use_gpu = gpu
        self.hub_model_name = None

        self.ssa_ref_file = ssa_ref_file
        self.seq_msa = seq_msa
        self.pool_residues = None
        self.ref_seq = None
        self.evotuned_dir = None
        self.pool_num = pool_num

        # Set pooling strategy and different parameters required
        self.pool_strat = pool_prot_strategy
        self.pool_residues_mapping = {}
        if self.pool_strat in ["ssa", "attn", "contact"]:
            assert self.ssa_ref_file is not None
            self.ref_seq, self.pool_residues = parse_utils.parse_ssa_reference(
                self.ssa_ref_file
            )
            self.pool_residues = sorted(self.pool_residues)
            self.ref_embedding = None
            if self.pool_strat in ["attn", "contact"]:
                assert isinstance(self, ESMFeaturizer)
        elif self.pool_strat in [""]:
            pass
        elif self.pool_strat in ["hard", "hardcat"]:
            assert self.ssa_ref_file is not None
            assert self.seq_msa is not None
            self.ref_seq, self.pool_residues = parse_utils.parse_ssa_reference(
                self.ssa_ref_file
            )
            self.pool_residues = sorted(self.pool_residues)
            self.pool_residues_mapping = parse_utils.extract_pool_residue_dict(
                self.seq_msa, self.ref_seq, self.pool_residues
            )
        elif self.pool_strat == "randhard":
            assert self.ssa_ref_file is not None
            assert self.seq_msa is not None
            self.ref_seq, self.pool_residues = parse_utils.parse_ssa_reference(
                self.ssa_ref_file
            )
            logging.info("Randomizing pool residues")
            seqlen = len(self.ref_seq)
            num_residues_chosen = len(self.pool_residues)
            new_positions = np.random.choice(
                a=np.arange(seqlen), replace=False, size=num_residues_chosen
            )
            self.pool_residues = new_positions
            self.pool_residues = sorted(self.pool_residues)
            self.pool_residues_mapping = parse_utils.extract_pool_residue_dict(
                self.seq_msa, self.ref_seq, self.pool_residues
            )
        elif self.pool_strat == "randmsa":
            assert self.seq_msa is not None
            logging.info("Randomizing pool residues")
            self.pool_residues_mapping = parse_utils.extract_pool_residue_dict(
                self.seq_msa, None, None, rand=True, pool_num=self.pool_num
            )
        elif self.pool_strat == "msacover":
            assert self.seq_msa is not None
            logging.info("Extracting pool residue positions")

            # Get the residues that have the most coverage
            self.pool_residues_mapping = parse_utils.extract_coverage_residue_dict(
                self.seq_msa, pool_num=self.pool_num
            )
        elif self.pool_strat == "msaconserv":
            assert self.seq_msa is not None
            logging.info("Extracting pool residue positions")
            # Get the residues that have the most coverage
            self.pool_residues_mapping = parse_utils.extract_conserve_residue_dict(
                self.seq_msa, pool_num=self.pool_num
            )
        else:
            self.pool_residues = None
            self.ref_embedding = None

    def out_dim(self) -> Tuple[Optional[int]]:
        """out_dim.

        Get the out dimension of this featurizer.
        If it's variable, return (None, hidden)

        Returns:
            Tuple[Optional[int]]:
        """
        if self.out_dim_set is not None:
            # If manually set
            # NOTE: This should not be an option
            return self.out_dim_set
        else:
            if len(self.feature_cache) > 0:
                outdim = [next(iter(self.feature_cache.values()))]
            else:
                # dummy amino acid list
                temp_seq = ["MSTNG"]
                outdim = self.featurize(temp_seq, use_cache=False)

            if self.pool_strat is not None:
                return (outdim[0].shape[0],)
            else:
                return (
                    None,
                    outdim[0].shape[1],
                )

    def fill_cache_get_remaining(
        self, seq_list: List[str], use_cache: bool, unpooled=False
    ) -> Tuple[dict, List[str]]:
        """fill_cache_get_remaining.

        Fill the cache if use_cache is true, and then get all the remaining
        sequence items

        Args:
            seq_list (List[str]): List of strings to check if they're in the
                cache
            use_cache (bool): If true, use the cache
            unpooled (bool): If true, use the unpooled cache
        Return:
            Tuple[dict, List[str]] Dict containing cache and list of remaining
            items
        """

        feature_cache = (
            self.feature_cache if not unpooled else self.unpooled_feature_cache
        )

        seq_list_mapping = dict()
        if use_cache:
            if unpooled:
                self.fill_unpooled_cache()
            else:
                self.fill_feature_cache()

        to_featurize = []
        for seq in seq_list:
            if seq in feature_cache:
                seq_list_mapping[seq] = feature_cache[seq]
            else:
                to_featurize.append(seq)

        return seq_list_mapping, to_featurize

    def fill_unpooled_cache(self):
        """fill_unpooled_cache.

        Fill the _unpooled_ cache for this protein.
        """
        model_cache_str = self.get_model_props_unpooled()
        feat_cache = self.unpooled_feature_cache
        if self.cache_dir and not model_cache_str:
            logging.info(
                "No method get_unpooled_props() implemented for featurizer {str(self)}"
            )
            return
        return self._fill_feature_cache(model_cache_str, feat_cache)

    def dump_features_unpooled(self):
        """dump_features_unpooled.

        Dump the _unpooled_ cache for this prot featurizer
        """
        model_cache_str = self.get_model_props_unpooled()
        self._dump_features(model_cache_str, self.unpooled_feature_cache)

    def forward_pass(
        self, seq_list: List[str]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """forward_pass.

        Compute a forward pass of the model. This abstracts away more complex
        logic with cacheing in the featurize call

        Args:
            seq_list (List[str]): List of seqs to embed

        Return:
            Tuple[List[np.ndarray], List[np.ndarray], List[int]]: Embeddings,
                interactions, and list of lengths
        """
        if self.use_gpu:
            self.model = self.model.cuda()

        batch = seq_list
        with torch.no_grad():
            #  NOTE: This keeps cls and separators for tape at the start
            #  and end
            encoded_list = [self.tokenizer.encode(seq) for seq in batch]
            input_mask = [np.ones(subset_el.shape) for subset_el in encoded_list]
            padded_input = tape_dataset.pad_sequences(encoded_list)
            padded_mask = tape_dataset.pad_sequences(input_mask)

            # Convert to torch
            padded_mask = torch.from_numpy(padded_mask)
            padded_input = torch.from_numpy(padded_input)

            if self.use_gpu:
                padded_mask = padded_mask.cuda()
                padded_input = padded_input.cuda()

            # run model
            sequence_output = self.model(padded_input)
            sequence_embeds = sequence_output[0].cpu().numpy()
            pooled_embeds = sequence_output[1].cpu().numpy()

            # Get seqlens
            seq_lens = padded_mask.sum(1).cpu().numpy()

        return (sequence_embeds, None, seq_lens)

    def pool_single_embedding(
        self,
        orig_seq: str,
        seq_embed: np.ndarray,
        seq_len: int,
        seq_attn: np.ndarray = None,
        seq_contacts: np.ndarray = None,
    ) -> np.ndarray:
        """pool_single_embedding.

        Pool a single embedding.

        Args:
            orig_seq (str): Original sequence
            seq_embed (np.ndarray): Embedding of original sequence
            seq_len (int): Length of sequence
            seq_attn (np.ndarray): Attention matrix of the sequence
            seq_contacts (np.ndarray): contact matrix of the sequence
        Return:
            np.ndarray: Size of new embedding
        """

        if self.pool_strat is None:
            return_embedding = seq_embed[:seq_len]
        elif self.pool_strat in [
            "hard",
            "randhard",
            "randmsa",
            "msacover",
            "msaconserv",
        ]:
            pool_residues = self.pool_residues_mapping[orig_seq]
            return_embedding = np.mean(seq_embed[pool_residues], axis=0)
            if np.any(np.isnan(return_embedding)):
                logging.info("Found NAN in embedding; converting to 0")
                return_embedding = np.nan_to_num(return_embedding)

        elif self.pool_strat in ["hardcat"]:
            pool_residues = self.pool_residues_mapping[orig_seq]
            return_embedding = np.hstack(seq_embed[pool_residues])
        elif self.pool_strat == "ssa":
            # Note this hsould be squeezed
            assert self.ref_embedding is not None
            return_embedding = ssa_utils.soft_pool_to_ref(
                seq_embed, self.ref_embedding, self.pool_residues
            )
        elif self.pool_strat == "mean":
            return_embedding = np.mean(seq_embed[:seq_len], axis=0)

        elif self.pool_strat in ["attn", "contact"]:
            # Pool based on reference embeddings and attention
            assert self.ref_embedding is not None

            # Do this on cuda now
            with torch.no_grad():
                ref_embedding = self.ref_embedding[self.pool_residues]
                pool_mat = seq_attn if self.pool_strat == "attn" else seq_contacts

                ref_embedding_torch = torch.from_numpy(ref_embedding)
                pool_mat_torch = torch.from_numpy(pool_mat)
                seq_embed_torch = torch.from_numpy(seq_embed)

                if self.use_gpu:
                    ref_embedding_torch = ref_embedding_torch.cuda()
                    pool_mat_torch = pool_mat_torch.cuda()
                    seq_embed_torch = seq_embed_torch.cuda()

                dist_mat = ssa_utils.l1_dist_matrix_torch(
                    seq_embed_torch, ref_embedding_torch
                )

            # Beta: seqlen x |A|
            beta_mat = ssa_utils.get_beta_torch(dist_mat)

            # Einsum magic
            # i : seq len of s1
            # j : seq len of s1
            # k : pool residues length
            # l : pool residues length
            # d : attention dimension
            # (i,k) x (i,i, d) x (k, i) =>  (k,k, d)
            prod_one = torch.einsum("ik,ijd->kjd", beta_mat, pool_mat_torch)
            prod_two = torch.einsum("jl,kjd->kld", beta_mat, prod_one)

            # Now flatten it all out to get one embedding
            return_embedding = prod_two.cpu().numpy().reshape(-1)

        else:
            raise NotImplementedError(f"Strategy {self.pool_strat}")

        return return_embedding

    def featurize(self, seq_list: List[str], use_cache=True) -> List[np.ndarray]:
        """featurize.

        On first settles on fixed kmer components

        Args:
            seq_list (List[str]): seq_list containing strings of smiles
            use_cache (bool): If false, don't use the cache

        Returns:
            List(np.ndarray): of features
        """

        seq_list_mapping, to_featurize = self.fill_cache_get_remaining(
            seq_list, use_cache
        )

        if self.use_gpu:
            self.model = self.model.cuda()

        # Add 1 to correct for the one item set case
        num_sections = (len(to_featurize) // self.embed_batch_size) + 1
        seq_list_sorted = sorted(to_featurize, key=len)
        batches = np.array_split(seq_list_sorted, num_sections)

        with torch.no_grad():
            for batch in tqdm(batches):

                # If we have nothing in the batch
                if len(batch) == 0:
                    break

                # Comput forward pass for this isngle batch
                (
                    sequence_embeds,
                    pooled_embeds,
                    seq_lens,
                ) = self.forward_pass(batch)

                # Now compute normalization
                for orig_seq, seq_embed, pooled_embed, seq_len in zip(
                    batch, sequence_embeds, pooled_embeds, seq_lens
                ):

                    # First embed the reference!
                    if self.pool_strat == "ssa" and self.ref_embedding is None:
                        ref_embed, ref_pool, ref_len = self.forward_pass([self.ref_seq])

                        # Dimension is len x embed_dim
                        self.ref_embedding = ref_embed[0].squeeze()

                    seq_list_mapping[orig_seq] = self.pool_single_embedding(
                        orig_seq, seq_embed, int(seq_len)
                    )

        output = [seq_list_mapping[seq] for seq in seq_list]

        # Dump features learned and update cache
        self.feature_cache.update(seq_list_mapping)
        self.dump_features()

        return list(output)

    def get_model_props_unpooled(self):
        """Use this to encode the model into a cache file"""
        raise NotImplementedError()

    def get_model_props(self):
        """Use this to encode the model into a cache file"""
        model_str = self.model.cpu().__str__()
        pooling_num = self.pool_strat
        model_str = {
            "pool_fn": pooling_num,
            "pool_num": self.pool_num,
            "model_desc": model_str,
            "pool_file": self.seq_msa,
            "evotuned_dir": self.evotuned_dir,
            "ssa_ref_file": self.ssa_ref_file,
        }
        if self.hub_model_name is not None:
            model_str["hub_model_name"] = self.hub_model_name
        hashed_str = file_utils.md5(str(model_str))
        return hashed_str


class BERTFeaturizer(TAPEFeaturizer):
    """BERTFeaturizer.

    Use TAPE to help featurize proteins.

    """

    def __init__(self, evotuned_dir: Optional[str] = None, **kwargs):
        """__init__.

        Args:
            evotuned_dir (str): Location of the evotuned file
            kwargs: kwargs
        """

        super().__init__(**kwargs)

        if not evotuned_dir:
            evotuned_dir = "bert-base"
        self.model = ProteinBertModel.from_pretrained(evotuned_dir)

        # iupac is the vocab for TAPE models, use unirep for the UniRep model
        self.tokenizer = TAPETokenizer(vocab="iupac")
        self.evotuned_dir = evotuned_dir


class UniRepFeaturizer(TAPEFeaturizer):
    """UniRepFeaturizer.

    Use TAPE to help featurize proteins.

    """

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """

        super().__init__(**kwargs)
        self.model = UniRepModel.from_pretrained("babbler-1900")
        self.tokenizer = TAPETokenizer(
            vocab="unirep"
        )  # iupac is the vocab for TAPE models, use unirep for the UniRep model


class BeplerFeaturizer(TAPEFeaturizer):
    """BeplerFeaturizer.

    Use Tristan Beplers 2019 ICLR model to featurize proteins.

    For now, we borrow  most of the methods from the TAPEFeaturizer, but
    override the featurize function

    Forked version of the code for install as package:
        https://github.com/samgoldman97/protein-sequence-embedding-iclr2019/tree/master/bepler_embedding

    """

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """

        # If false, fit, otherwise just run
        super().__init__(**kwargs)

        # Load pretrained
        logging.info(f"Creating bepler representation model")
        self.model = be.get_default_saved_model()
        logging.info(f"Done bepler representation model")

    def featurize(self, seq_list: List[str], use_cache=True) -> List[np.ndarray]:
        """featurize.

        On first settles on fixed kmer components

        Args:
            seq_list (List[str]): seq_list containing strings of smiles
            use_cache (bool): If false, don't use the cache

        Returns:
            List(np.ndarray): of features
        """

        seq_list_mapping, to_featurize = self.fill_cache_get_remaining(
            seq_list, use_cache
        )

        if self.use_gpu:
            self.model = self.model.cuda()

        embeddings = be.embed_sequences(
            self.model, to_featurize, self.embed_batch_size, self.use_gpu
        )

        # First embed the reference!
        if self.pool_strat == "ssa" and self.ref_embedding is None:
            ref_embed = be.embed_sequences(
                self.model, [self.ref_seq], self.embed_batch_size, self.use_gpu
            )[0]
            # Dimension is len x embed_dim
            self.ref_embedding = ref_embed.squeeze()

        # Pool all of these!
        embeddings = [
            self.pool_single_embedding(orig_seq, embedding, int(len(orig_seq)))
            for orig_seq, embedding in zip(to_featurize, embeddings)
        ]

        # Update with new feature additions
        seq_list_mapping.update(dict(zip(to_featurize, embeddings)))

        output = [seq_list_mapping[seq] for seq in seq_list]

        # Dump features learned and update cache
        self.feature_cache.update(seq_list_mapping)
        self.dump_features()
        return list(output)


class ESMFeaturizer(TAPEFeaturizer):
    """BeplerFeaturizer.

    Use Facebook's Evolutionary Scale Model to featurize proteins.

    For now, we borrow  most of the methods from the TAPEFeaturizer, but
    override the featurize function

    """

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """

        # If false, fit, otherwise just run
        super().__init__(**kwargs)

        # Load pretrained
        logging.info(f"Creating esm representation model")

        # Grab the file name so we can differentiate caches for memory
        self.hts_file = kwargs.get("hts_csv_file", None)
        if self.hts_file is not None:
            self.hts_file = self.hts_file.split("/")[-1].split("_")[0]

        self.hub_model_name = "esm1b_t33_650M_UR50S"  # "esm1_t34_670M_UR50S"  #

        self.model, self.alphabet = None, None

        self.repr_layer = 34 if self.hub_model_name == "esm1_t34_670M_UR50S" else 33
        logging.info(f"Done esm representation model")

    def forward_pass(
        self, seq_list: List[str]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[np.ndarray]]:
        """forward_pass.

        Compute a forward pass of the model. This abstracts away more complex
        logic with caching in the featurize call

        Args:
            seq_list (List[str]): List of seqs to embed

        Return:
            Tuple[List[np.ndarray], List[np.ndarray], List[int], List[np.ndarray]]: Embeddings,
                attentions, list of lengths, contacts
        """
        if self.model is not None and self.use_gpu:
            self.model = self.model.cuda()

        with torch.no_grad():
            data_batch = [(i, i) for i in seq_list]
            seqlens = [len(i) for i in seq_list]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data_batch)

            if self.use_gpu:
                batch_tokens = batch_tokens.cuda()

            results = self.model(
                batch_tokens,
                repr_layers=[self.repr_layer],
                return_contacts=self.pool_strat in ["attn", "contact"],
            )

            # Handle contacts
            if self.pool_strat in ["attn", "contact"]:
                attns = results["attentions"].cpu().numpy()
                contacts = results["contacts"].cpu().numpy()
                num_items = attns.shape[0]
                length_2 = attns.shape[-1]
                length_1 = attns.shape[-2]

                # Correct for a bug in the code for facebook with the ESM1
                # model where they don't remove the extra token
                assert length_1 == length_2

                # Collapse attention dimension and num layers into one
                attns = attns.reshape((num_items, -1, length_1, length_1))

                # Trapose to be (batch, length, length, dim)
                attns = attns.transpose((0, 2, 3, 1))

                # Reshape contacts accordingly
                contacts = contacts.reshape((num_items, length_1 - 2, length_2 - 2, 1))

                # now reduce to length x length
                attn_list = [
                    i[1 : len(seq) + 1, :][:, 1 : len(seq) + 1]
                    for seq, i in zip(seq_list, attns)
                ]
                contact_list = [
                    i[: len(seq), :][:, : len(seq)]
                    for seq, i in zip(seq_list, contacts)
                ]
            else:
                attn_list = [None for seq in seq_list]
                contact_list = [None for seq in seq_list]

            token_embeddings = results["representations"][self.repr_layer]
            token_embeddings = token_embeddings.cpu().numpy()

        # Remove the start token!
        token_embeddings_list = [
            i[1 : len(seq) + 1] for seq, i in zip(seq_list, token_embeddings)
        ]
        return (token_embeddings_list, attn_list, seqlens, contact_list)

    def featurize(self, seq_list: List[str], use_cache=True) -> List[np.ndarray]:
        """featurize.

        On first settles on fixed kmer components

        Args:
            seq_list (List[str]): seq_list containing strings of smiles
            use_cache (bool): If false, don't use the cache

        Returns:
            List(np.ndarray): of features
        """
        seq_list_mapping, to_featurize = self.fill_cache_get_remaining(
            seq_list, use_cache
        )

        # Get a secondary cache if we have to featurize more proteins
        seq_list_mapping_unpooled, to_featurize_unpooled = {}, []
        if len(to_featurize) > 0:
            (
                seq_list_mapping_unpooled,
                to_featurize_unpooled,
            ) = self.fill_cache_get_remaining(seq_list, use_cache, unpooled=True)

        # Only fetch the model if we have to
        if len(to_featurize_unpooled) > 0 and self.model is None:
            self.model, self.alphabet = torch.hub.load(
                "facebookresearch/esm", self.hub_model_name
            )
            self.batch_converter = self.alphabet.get_batch_converter()

        if self.model is not None and self.use_gpu:
            self.model = self.model.cuda()

        ### Run featurization routine
        num_splits = (len(to_featurize_unpooled) // self.embed_batch_size) + 1
        embedding_list, attn_list, seqlen_list, sequence_list, contact_list = (
            [],
            [],
            [],
            [],
            [],
        )
        with torch.no_grad():
            # collect all unpooled embeddings
            for data_batch in tqdm(np.array_split(to_featurize_unpooled, num_splits)):

                if len(data_batch) == 0:
                    break

                token_embeddings, seq_attns, seqlens, seq_contact = self.forward_pass(
                    data_batch
                )

                # embedding_list.extend(token_embeddings)
                # attn_list.extend(seq_attns)
                # contact_list.extend(seq_contact)
                # seqlen_list.extend(seqlens)
                # sequence_list.extend(data_batch)

                # Update unpooled cache here
                for seq_item, embedding_item in zip(data_batch, token_embeddings):
                    seq_list_mapping_unpooled[seq_item] = embedding_item

        # Embed the reference before pooling!
        # Note: with new strategy, we don't bother with a reference
        if self.pool_strat in ["ssa", "attn", "contact"] and self.ref_embedding is None:
            # TODO: Now that we've restructured this code, we would need to
            # pull down the model again here
            raise NotImplementedError()
            # ref_embed, ref_pool, ref_len, ref_contacts = self.forward_pass([self.ref_seq])

            ## Dimension is len x embed_dim
            # self.ref_embedding = ref_embed[0].squeeze()

        logging.info("Starting to pool ESM Embeddings")

        # Collect all embeddings from the unpooled embedding mapping
        embedding_list = [seq_list_mapping_unpooled[seq] for seq in to_featurize]
        sequence_list = to_featurize
        attn_list = [None for _ in to_featurize]
        contact_list = [None for _ in to_featurize]
        seqlen_list = [len(seq) for seq in to_featurize]

        # Pool all of these!
        embeddings = [
            self.pool_single_embedding(
                orig_seq, embedding, seqlen, seq_attn=seq_attn, seq_contacts=contacts
            )
            for orig_seq, embedding, seq_attn, seqlen, contacts in tqdm(
                zip(sequence_list, embedding_list, attn_list, seqlen_list, contact_list)
            )
        ]
        # Update with new feature additions
        seq_list_mapping.update(dict(zip(to_featurize, embeddings)))

        # Note: We updated the unpooled version dynamically
        output = [seq_list_mapping[seq] for seq in seq_list]

        # Dump features learned and update cache
        self.feature_cache.update(seq_list_mapping)
        self.dump_features()

        # Also dump the unpooled features if we had to add any new ones
        if len(to_featurize_unpooled) > 0:
            self.unpooled_feature_cache.update(seq_list_mapping_unpooled)
            self.dump_features_unpooled()

        return list(output)

    def _get_model_prop_dict(self):
        """Use this to encode the model into a cache file"""
        pooling_num = self.pool_strat

        # Record msa pool positions
        pooled_residues = file_utils.md5(str(self.pool_residues_mapping))
        model_dict = {
            "pool_fn": pooling_num,
            "pool_num": self.pool_num,
            "pool_file": self.seq_msa,
            "evotuned_dir": self.evotuned_dir,
            "pooled_residues": pooled_residues,
            "ssa_ref_file": self.ssa_ref_file,
        }

        if self.hub_model_name is not None:
            model_dict["hub_model_name"] = self.hub_model_name
        return model_dict

    def get_model_props_unpooled(self):
        """Use this to encode the model into a cache file"""
        model_str = self._get_model_prop_dict()
        model_str["pool_fn"] = None
        model_str["pool_num"] = None
        model_str["pool_file"] = None
        model_str["ssa_ref_file"] = None
        model_str["pooled_residues"] = None
        model_str["dataset"] = self.hts_file
        hashed_str = file_utils.md5(str(model_str))
        return hashed_str

    def get_model_props(self):
        """Use this to encode the model into a cache file"""
        model_str = self._get_model_prop_dict()
        hashed_str = file_utils.md5(str(model_str))
        return hashed_str


class KMERFeaturizer(ProtFeaturizer):
    """KMERFeaturizer.

    Implement kmer featurizer

    """

    def __init__(
        self,
        ngram_min: int = 2,
        ngram_max: int = 4,
        unnormalized: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
            ngram_min (int): ngram_min
            ngram_max (int): ngram_max
            unnormalized (bool): normalize
            kwargs: kwargs
        """

        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.normalize = not (unnormalized)

        self.vectorizer = CountVectorizer(
            ngram_range=(self.ngram_min, self.ngram_max), analyzer="char"
        )

        # If false, fit, otherwise just run
        self.is_fit = False

        super().__init__(**kwargs)

    def out_dim(self) -> Tuple[Optional[int]]:
        """out_dim.

        Get the out dimension of this featurizer.
        If it's variable, return None

        Returns:
           Tuple[Optional[int]]
        """
        if not self.is_fit:
            return (None,)
        elif self.out_dim_set is not None:
            return self.out_dim_set
        else:
            return (len(self.vectorizer.get_feature_names()),)

    def featurize(self, seq_list: List[str]) -> List[np.ndarray]:
        """featurize.

        On first settles on fixed kmer components

        Args:
            seq_list (List[str]): seq_list containing strings of smiles

        Returns:
            np.ndarray: of features
        """
        if not self.is_fit:
            self.vectorizer.fit(seq_list)
            self.is_fit = True
        output = self.vectorizer.transform(seq_list)
        output = np.asarray(output.todense())

        # If this is true, normalize the sequence
        if self.normalize:
            output = output / output.sum(1).reshape(-1, 1)

        return list(output)


class StrFeaturizer(ProtFeaturizer):
    """StrFeaturizer.

    Return the str of the protein. Use this if the model has its own batch
    converter

    """

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """
        super().__init__(**kwargs)

    def out_dim(self) -> Tuple[Optional[int]]:
        """out_dim.

        Get the out dimension of this featurizer.
        If it's variable, return None

        Returns:
           Tuple[Optional[int]]
        """
        return (None,)

    def featurize(self, seq_list: List[str]) -> List[str]:
        """featurize.

        Args:
            seq_list (List[str]): seq_list containing strings of smiles

        Returns:
            Str list
        """

        # Encoded seqs
        return seq_list


class Uniprot21Featurizer(ProtFeaturizer):
    """Uniprot21Featurizer.

    One hot featurize a protein

    """

    def __init__(self, **kwargs):
        """__init__.

        Args:
            kwargs: kwargs
        """

        self.alphabet = alphabet.Uniprot21()
        self.outsize = self.alphabet.output_size
        self.ident = np.identity(self.outsize)
        super().__init__(**kwargs)

    def out_dim(self) -> Tuple[Optional[int]]:
        """out_dim.

        Get the out dimension of this featurizer.
        If it's variable, return None

        Returns:
           Tuple[Optional[int]]
        """
        return (None, self.alphabet.output_size)

    def featurize(self, seq_list: List[str]) -> List[np.ndarray]:
        """featurize.

        Args:
            seq_list (List[str]): seq_list containing strings of smiles

        Returns:
            np.ndarray: of one hot encoded features
        """

        # Encoded seqs
        output = [self.ident[self.alphabet.encode(seq)] for seq in seq_list]

        return output


class Uniprot21FeaturizerMSA(ProtFeaturizer):
    """Uniprot21FeaturizerMSA.

    One hot featurize proteins based on a subset of residues from an MSA

    """

    def __init__(self, ssa_ref_file: str, seq_msa: str, **kwargs):
        """__init__.

        Args:
            ssa_ref_file (str): Name of reference SSA file
            seq_msa (str): Sequence MSA
            kwargs: kwargs
        """

        self.alphabet = alphabet.Uniprot21()
        self.outsize = self.alphabet.output_size
        self.ident = np.identity(self.outsize)

        self.ssa_ref_file = ssa_ref_file
        self.seq_msa = seq_msa
        self.ref_seq, self.pool_residues = parse_utils.parse_ssa_reference(
            self.ssa_ref_file
        )
        self.pool_residues = sorted(self.pool_residues)

        self.msa_map = {
            seq.replace("-", ""): seq
            for seq_name, seq in parse_utils.fasta_iter(seq_msa)
        }

        ref_aligned_mapping = parse_utils.map_aligned(self.msa_map[self.ref_seq])
        # Set of all positions in the alignment we should pool over
        self.alignment_pool_positions = [
            ref_aligned_mapping[j] for j in self.pool_residues
        ]

        super().__init__(**kwargs)

    def out_dim(self) -> Tuple[Optional[int]]:
        """out_dim.

        Get the out dimension of this featurizer.
        If it's variable, return None

        Returns:
           Tuple[Optional[int]]
        """
        return (len(self.pool_residues) * self.alphabet.output_size,)

    def featurize(self, seq_list: List[str]) -> List[np.ndarray]:
        """featurize.

        Args:
            seq_list (List[str]): seq_list containing strings of smiles

        Returns:
            np.ndarray: of one hot encoded features
        """
        # outputs
        embedding_list = []
        for unaligned_seq in seq_list:
            if unaligned_seq not in self.msa_map:
                raise ValueError(f"Seq {seq} not found in msa {self.seq_msa}")

            aligned_seq = self.msa_map[unaligned_seq]
            embedding = []
            for position in self.pool_residues:
                if aligned_seq[position] == "-":
                    embedding.append(np.zeros(self.alphabet.output_size))
                else:
                    embedding.append(
                        self.ident[
                            self.alphabet.encode(aligned_seq[position])
                        ].squeeze()
                    )
            embedding_list.append(np.vstack(embedding).flatten())

        return embedding_list


class MSAFeaturizer(ProtFeaturizer):
    """MSAFeaturizer.

    Return a sequence of proteins with gaps

    """

    def __init__(self, ssa_ref_file: str, seq_msa: str, **kwargs):
        """__init__.

        Args:
            ssa_ref_file (str): Name of reference SSA file
            seq_msa (str): Sequence MSA
            kwargs: kwargs
        """

        self.alphabet = alphabet.Uniprot21()
        self.outsize = self.alphabet.output_size
        self.ident = np.identity(self.outsize)

        self.ssa_ref_file = ssa_ref_file
        self.seq_msa = seq_msa
        self.ref_seq, self.pool_residues = parse_utils.parse_ssa_reference(
            self.ssa_ref_file
        )
        self.pool_residues = sorted(self.pool_residues)

        self.msa_map = {
            seq.replace("-", ""): seq
            for seq_name, seq in parse_utils.fasta_iter(seq_msa)
        }

        ref_aligned_mapping = parse_utils.map_aligned(self.msa_map[self.ref_seq])
        # Set of all positions in the alignment we should pool over
        self.alignment_pool_positions = [
            ref_aligned_mapping[j] for j in self.pool_residues
        ]

        super().__init__(**kwargs)

    def out_dim(self) -> Tuple[Optional[int]]:
        """out_dim.

        Get the out dimension of this featurizer.
        If it's variable, return None

        Returns:
           Tuple[Optional[int]]
        """
        return (len(self.pool_residues),)

    def featurize(self, seq_list: List[str]) -> List[np.ndarray]:
        """featurize.

        Args:
            seq_list (List[str]): seq_list containing strings of smiles

        Returns:
            np.ndarray: of one hot encoded features
        """
        # outputs
        embedding_list = []
        for unaligned_seq in seq_list:
            if unaligned_seq not in self.msa_map:
                raise ValueError(f"Seq {seq} not found in msa {self.seq_msa}")

            aligned_seq = self.msa_map[unaligned_seq]
            embedding = [aligned_seq[position] for position in self.pool_residues]

            embedding_list.append("".join(embedding))

        return embedding_list


class FullMSAFeaturizer(ProtFeaturizer):
    """MSAFeaturizer.

    Return a

    """

    def __init__(self, ssa_ref_file: str, seq_msa: str, **kwargs):
        """__init__.

        Args:
            ssa_ref_file (str): Name of reference SSA file
            seq_msa (str): Sequence MSA
            kwargs: kwargs
        """

        self.alphabet = alphabet.Uniprot21()
        self.outsize = self.alphabet.output_size
        self.ident = np.identity(self.outsize)
        self.msa_map = {
            seq.replace("-", ""): seq
            for seq_name, seq in parse_utils.fasta_iter(seq_msa)
        }
        self.out_dim_ = len(list(self.msa_map.values())[0])

        super().__init__(**kwargs)

    def out_dim(self) -> Tuple[Optional[int]]:
        """out_dim.

        Get the out dimension of this featurizer.
        If it's variable, return None

        Returns:
           Tuple[Optional[int]]
        """
        return (self.out_dim_,)

    def featurize(self, seq_list: List[str]) -> List[np.ndarray]:
        """featurize.

        Args:
            seq_list (List[str]): seq_list containing strings of smiles

        Returns:
            np.ndarray: of one hot encoded features
        """

        # outputs
        embedding_list = []
        for unaligned_seq in seq_list:
            if unaligned_seq not in self.msa_map:
                raise ValueError(f"Seq {seq} not found in msa {self.seq_msa}")

            aligned_seq = self.msa_map[unaligned_seq]

            # Note that gap is 20
            embedding = self.alphabet.encode(aligned_seq)
            embedding_list.append(embedding)

        return embedding_list
