"""Module containing some helper functions

    Typical usage example:

    from enzpred.utils import file_utils
    file_utils.make_dir("foo/bar/baz.txt")
"""
import pickle
from typing import Any
import os
import json
import logging
import hashlib


def md5(key: str) -> str:
    """md5.

    Args:
        key (str): string to be hasehd
    Returns:
        Hashed encoding of str
    """
    return hashlib.md5(key.encode()).hexdigest()


def dump_json(
    obj: dict, outfile: str = "temp_dir/temp.json", pretty_print: bool = True
) -> None:
    """pickle_obj.

    Helper fn to pickle object

    Args:
        obj (Any): dict
        outfile (str): outfile
        pretty_print (bool): If true, use tabs

    Returns:
        None
    """
    if pretty_print:
        json.dump(obj, open(outfile, "w"), indent=2)
    else:
        json.dump(obj, open(outfile, "w"))


def load_json(infile: str = "temp_dir/temp.p") -> Any:
    """load_json.

    Args:
        infile (str): infile, the name of input object

    Returns:
        Any: the object loaded from pickled file

    """

    with open(infile, "r") as fp:
        return json.load(fp)


def pickle_obj(obj: Any, outfile: str = "temp_dir/temp.p") -> None:
    """pickle_obj.

    Helper fn to pickle object

    Args:
        obj (Any): obj
        outfile (str): outfile

    Returns:
        None
    """
    with open(outfile, "wb") as fp:
        pickle.dump(obj, fp)


def pickle_load(infile: str = "temp_dir/temp.p") -> Any:
    """pickle_load.

    Args:
        infile (str): infile, the name of input object

    Returns:
        Any: the object loaded from pickled file

    """
    with open(infile, "rb") as fp:
        return pickle.load(fp)


def make_dir(filename: str) -> None:
    """make_dir.

    Makes the directory that should contain this file

    Args:
        filename (str): filename

    Returns:
        None
    """
    # Make outdir if it doesn't exist
    out_folder = os.path.dirname(filename)
    os.makedirs(out_folder, exist_ok=True)


class Stage(object):
    """Stage.

    Class to handle logger formatting

    Taken from: https://github.com/connorcoley/MolPAL/blob/master/molpal/utils.py

    TODO: Move this to a logger utils file

    """

    def __init__(self, label):
        """__init__.

        Args:
            label:
        """
        self.label = label
        self.logger = logging.getLogger()
        self.base_fmt = self.logger.handlers[0].formatter._fmt
        logging.info(f"Starting stage: {self.label}")

        for handler in self.logger.handlers:
            handler.setFormatter(
                logging.Formatter(self.base_fmt.replace("%(message)s", "  %(message)s"))
            )

    def __enter__(self):
        """__enter__."""
        pass

    def __exit__(self, type, value, traceback):
        """__exit__.

        Args:
            type:
            value:
            traceback:
        """

        for handler in self.logger.handlers:
            handler.setFormatter(logging.Formatter(self.base_fmt))

        logging.info(f"Done with stage: {self.label}")
