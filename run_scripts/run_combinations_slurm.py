""" Pythonic launcher

Use combinations config file to launch a bnch of trials

python run_scripts/run_combinations_slurm.py configs/scratch_file_combinations.json

To use this script, the config file should be structured using a set of
launcher args, universal args, and then an iterable iterative args that will
override the universal args. 

Every parameter in the universal args section should be an argument to the
python file and contained iwhtin a list. The product of all these entries will
be computed and directly fed as arguments in different program calls.

Any argument beginning with an underscore is manipulated by this launcher file
and transformed before being fed into the corresponding program. 

    launcher_args: {experiment_name : str,
                    cluster_script : bool ,
                    slurm_script : strenzpred/scripts/generic_slrum.sh,
                    use_slurm : bool}
    universal_args: {program-arg-1 : [combo_1, combo_2],
                     program-arg-2: [combo_1, combo_2, combo_3],
                     ...
                     _dataset: [dataset_1, dataset_2, datset_2],
                     _angstrom_dists : [5,8,10],
                     _slurm_args : [{time: time, _num_gpu : 1, job-name}]
    }

    iterative_args: [{universal_arg_replacements_1},
                     {universal_arg_replacements_2}
                     ...]

"""

import os
import shutil
import hashlib
import subprocess 
import copy
import time 
import itertools
from datetime import datetime
import argparse
import json

def md5(key: str) -> str:
    """md5.

    Args:
        key (str): string to be hasehd
    Returns:
        Hashed encoding of str
    """
    return hashlib.md5(key.encode()).hexdigest()

def get_args(): 
    """ parse json config"""
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Name of configuration file")
    args = parser.parse_args()
    print(f"Loading experiment from: {args.config_file}\n")
    args_new = json.load(open(args.config_file, "r"))
    return args_new, args.config_file

def dump_config_file(save_dir : str, config : str): 
    """ dump_config_file.

    Try to dump the output config file continuously. If it doesn't work,
    increment it.  

    Args:
        save_dir (str): Name of the save dir where to put this
        config (str): Location of the config file
    """

    # Dump experiment
    new_file = "experiment.config"
    config_path = os.path.join(save_dir, new_file)
    ctr = 1
    os.makedirs(save_dir, exist_ok=True)

    # Keep incrementing the counter
    while os.path.exists(config_path): 
        new_file =  f"experiment_{ctr}.json"
        config_path = os.path.join(save_dir, new_file)
        ctr += 1

    shutil.copy2(config, config_path)
    time_stamp_date = datetime.now().strftime("%m_%d_%y")

    # Open timestamps file
    with open(os.path.join(save_dir, "timestamps.txt"), "a") as fp: 
          fp.write(f"Experiment {new_file} run on {time_stamp_date}.\n")


def build_python_string(experiment_folder : str, experiment_name, 
                        arg_dict : dict, launcher_args :dict): 
    """build_python_string. 
    """

    # python string
    script = "train_model.py" 

    python_string = f"python {script}"

    for arg_name, arg_value in arg_dict.items(): 
        if arg_name == "_slurm_args": 
            slurm_args = construct_slurm_args(experiment_name,  arg_value)
        elif arg_name == "_angstrom_dist" and arg_value is not None:
            dataset_prefix = arg_dict['_dataset'].split(".")[0].split("_")[0]
            prot_ref_file = f"data/processed/structure_references/{dataset_prefix}_reference_{arg_value}.txt"
            python_string = f"{python_string} --ssa-ref-file {prot_ref_file}".strip()
        elif arg_name == "_dataset": 
            dataset = arg_value
            dataset_args = f"--hts-csv-file data/processed/{dataset}.csv" 
            if "binary" in dataset or "categorical" in dataset: 
                dataset_args = f"{dataset_args}"
            else: 
                dataset_args = f"{dataset_args} --regression"

            dataset_prefix = dataset.split(".")[0].split("_")[0]
            msa_file = f"data/processed/alignments/{dataset_prefix}_alignment.fasta"
            sub_cat_file = f"data/processed/substrate_categories/{dataset_prefix}_sub_cats.p"
            dataset_args = f"{dataset_args} --seq-msa {msa_file} --substrate-cats-file {sub_cat_file}"

            python_string = f"{python_string} {dataset_args}".strip()

        else: 
            new_flag = convert_flag(arg_name, arg_value) 
            python_string = f"{python_string} {new_flag}".strip()

    # Define OUT 
    time_stamp_seconds = datetime.now().strftime("%Y_%m_%d-%H%M%S")
    time.sleep(1.2)

    dataset = arg_dict['_dataset']
    model = arg_dict['model']
    sub_dir = os.path.join(experiment_folder, dataset)
    sub_dir = os.path.join(sub_dir, model)
    args_hash = md5(str(arg_dict))  

    sub_dir = os.path.join(sub_dir, args_hash)
    out = os.path.join(sub_dir, time_stamp_seconds)
    python_string = f"{python_string} --out {out}"

    return (slurm_args, python_string)

def construct_slurm_args(experiment_name : str, slurm_args : dict): 
    """ construct_slurm_args."""

    # Slurm args 
    sbatch_args = f"--output=logs/{experiment_name}_%j.log"
    for k,v in slurm_args.items(): 
        if k == "_num_gpu": 
            if v > 0: 
                sbatch_args = f"{sbatch_args} --gres=gpu:volta:{v}"
        else: 
            new_flag = convert_flag(k, v) 
            sbatch_args = f"{sbatch_args} {new_flag}".strip()
    return sbatch_args

def convert_flag(flag_key, flag_value): 
    """ Convert the actual key value pair into a python flag"""
    if isinstance(flag_value, bool): 
        return_string = f"--{flag_key}" if flag_value else ""
    elif flag_value is None: 
        return_string =  "" 
    else: 
        return_string = f"--{flag_key} {flag_value}"
    return return_string

def main(config_file : str, launcher_args : list, 
         universal_args : dict, iterative_args : dict): 
    """ main. """
    # Create output experiment
    os.makedirs("logs", exist_ok=True)

    experiment_name = launcher_args['experiment_name']
    experiment_folder = f"results/dense/{experiment_name}/"
    dump_config_file(experiment_folder, config_file)

    ### Create launcher log
    launcher_path = os.path.join(experiment_folder, "launcher_log_1.log")

    # Keep incrementing the counter
    ctr = 1
    while os.path.exists(launcher_path): 
        new_file =  f"launcher_log_{ctr}.log"
        launcher_path = os.path.join(experiment_folder, new_file)
        ctr += 1
    log = open(launcher_path, "w")

    # List of current executions to run
    experiment_list = []
    # Loop over major arguments
    for arg_sublist in iterative_args:

        # Update universal args with iterative args
        # This overrides universal args with specific ones
        base_args = copy.copy(universal_args)
        base_args.update(arg_sublist)

        #  Now create combinations with updated list
        key, values  = zip(*base_args.items())
        combos = [dict(zip(key, val_combo))  
                  for val_combo in itertools.product(*values)]
        experiment_list.extend(combos)

    program_strs = []
    num_trials_complete=0
    for experiment_args in experiment_list: 
        program_strs.append(build_python_string(experiment_folder, experiment_name, 
                                                experiment_args, launcher_args))
    # Now actually launch programs
    for sbatch_args, python_str in program_strs: 
        if launcher_args.get("use_slurm", False):
            slurm_script = launcher_args.get("slurm_script",
                                             "enzpred/scripts/generic_slurm.sh")
            cmd_str = f"sbatch --export=CMD=\"{python_str}\" {sbatch_args} {slurm_script}"
            log.write(cmd_str + "\n")
            time.sleep(2.1)
        else: 
            cmd_str = python_str

        print(f"Command String: ", cmd_str)
        subprocess.call(cmd_str, shell=True)
        log.write(cmd_str+ "\n")
        num_trials_complete += 1

    log.close()

if __name__=="__main__": 
    os.makedirs("logs", exist_ok=True)
    args, config_file = get_args()
    main(config_file = config_file, **args)


