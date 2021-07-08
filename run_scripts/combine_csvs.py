"""
    consolidate_results.py
    
    Usage: 
    # No other files:
    python run_scripts/consolidate_results.py --results-dir results/test/ --out-file temp.csv

    # Other files:
    python run_scripts/consolidate_results.py --results-dir results/test/ --out-file temp.csv --other-csv-files temp.csv next.csv

"""
import os
import argparse
from typing import List
import pandas as pd
import json

import shutil

def get_args():
    """ Get arguments"""
    options = argparse.ArgumentParser()
    options.add_argument("--results-dir",
                         action="store",
                         help="Dir with folders that have *.csv files",
                         required=True)
    options.add_argument("--other-csv-files",
                         action="store",
                         nargs="*",
                         default=None,
                         help="Other csv files to merge")
    options.add_argument("--out-file",
                         action="store",
                         required=True,
                         help="Name of consolidated CSV")
    options.add_argument("--suffix",
                         action="store",
                         help="Name of CSV suffix",
                         default="output.csv")
    # Name of python file to run
    return options.parse_args()

def merge_files(sup_dir: str, suffix: str, other_csv_files: List[str],
                out_file: str):  
    """merge_files.

    Args:
        sup_dir (str): Directory containing reuslts of merge
        suffix (str): suffix of all files to merge
        other_csv_files (List[str]): List of other csv files to merge
        out_file (str): out_file name
    """
    subfolders = [f.path for f in os.scandir(sup_dir) if f.is_dir()]
    # Also include main dir
    subfolders.append(sup_dir)

    files_to_merge = []

    # Search each subfolder for .csv file
    #for subfolder in subfolders:
    #    csv_files = [
    #        j.path for j in os.scandir(subfolder) if j.path.endswith(suffix)
    #    ]

    #    files_to_merge.extend(csv_files)

    for (dirpath, dirnames, filenames) in os.walk(sup_dir):
        files_to_merge.extend(
            [os.path.join(dirpath, file) for file in filenames if file.endswith(suffix)]
        )


    # get other files
    other_files = other_csv_files
    if other_files is not None:
        files_to_merge.extend(other_files)


    # Get dfs
    df_list = []
    for df_file in files_to_merge:
        temp = pd.read_csv(df_file, index_col=0)
        # sort
        temp = temp.reindex(sorted(temp.columns), axis=1)
        df_list.append(temp)

    out_df = pd.concat(df_list, ignore_index=True)
    out_df.to_csv(out_file)

if __name__ == "__main__":
    args = get_args()
    sup_dir = args.results_dir

    merge_files(args.results_dir, args.suffix, args.other_csv_files,
                args.out_file) 
