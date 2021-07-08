"""This is a class to hold an alphabet for one-hot encoding proteins. 

This class inspired by the Bepler et al. representation learning paper.
"""
import numpy as np

# Conversion table (taken from Bepler et al.)
RESTABLE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "ASX": "B",
    "XLE": "J",
    "GLX": "Z",
    "UNK": "X",
    "XAA": "X",
}
PROTEIN_ALPHABET = b"ARNDCQEGHILKMFPSTWYVXOUBZ"
DNA_ALPHABET = b"ATGC"


class Alphabet:
    """Super class to hold alphabets"""

    def __init__(
        self,
        chars: bytes,
        encoding: np.ndarray = None,
        unk: int = None,
        start: int = None,
        end: int = None,
        pad: int = None,
    ):
        """Alphabet.

        Args:
            chars (byte): Byte sequence representing each char in utf8
            encoding (np.ndarray): mapping of those chars to ints; must be in
                numerical order s.t.  max(encoding) + 1 is the number of
                possible input integers
            unk (int) : Index of UNK token in chars;
                defaults to max(encoding) + 2 AND '_'
            start (int): Index of start token in chars;
                defaults to max(encoding) + 3 and b'<'
            stop (int): Index of stop token in chars;
                defaults to max(encoding) + 4 and b'>'
            pad (int): Index of pad token in chars;
                defaults to max(encoding) + 5 and b'0'

        """
        self.chars = np.frombuffer(chars, dtype=np.uint8)
        self.encoding = np.zeros(256, dtype=np.uint8)

        # Size of outputs for this model
        if encoding is not None:
            self.output_size = np.max(encoding) + 1
        else:
            self.output_size = len(chars)

        additional_indices = [unk, start, end, pad]
        new_defaults = [b"_"[0], b"<"[0], b">"[0], b"0"[0]]
        new_chars = []

        for index, default in zip(additional_indices, new_defaults):
            # Add UNK token to all defaults
            if index is not None:
                new_chars.append(index)
            else:
                # Verify default not already there..
                assert default not in self.chars
                self.chars = np.append(self.chars, default)
                # If encoding is not none, update it
                # Get index of added token
                new_chars.append(len(self.chars) - 1)

                if encoding is not None:
                    encoding = np.append(encoding, new_chars[-1])

        self.chars = self.chars.astype(np.uint8)

        self.unk = new_chars[0]
        self.start = new_chars[1]
        self.end = new_chars[2]
        self.pad = new_chars[3]

        # Make encoding default unk token
        self.encoding += self.unk

        # Make encoding using chars
        if encoding is None:
            self.encoding[self.chars] = np.arange(len(self.chars))
        else:
            self.encoding[self.chars] = encoding

        # Set size to be this + 1
        # Note: This will include start, unk, end, pad
        self.size = self.encoding.max() + 1

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i) -> chr:
        return chr(self.chars[i])

    def encode(self, x: str) -> np.ndarray:
        """Encode a str into alphabet indices"""
        x = np.frombuffer(x.encode(), dtype=np.uint8)
        return self.encoding[x]

    def decode(self, x: np.ndarray) -> str:
        """Decode index array, x, to string of this alphabet"""
        x = x.astype(np.uint8)
        string = self.chars[x]
        return string.tobytes().decode()


class Uniprot21(Alphabet):
    """Standard uniprot21 alphabet.

    This also includes characters to handle both UNK tokens and the synonymous
    B and Z residues.

    """

    def __init__(self):
        chars = alphabet = PROTEIN_ALPHABET
        encoding = np.arange(len(chars))

        # encode 'OUBZ' as synonyms where 'BZ' are ``missing" tokens
        encoding[21:] = [11, 4, 20, 20]
        super(Uniprot21, self).__init__(chars, encoding=encoding, unk=20)
