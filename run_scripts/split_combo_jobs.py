"""split_combo_jobs.py

Run a few different configs with the launcher on slurm

Take a set of jobs that are supposed to be dense and split them up such that
one node runs all the jobs for a single iteration as listed. 
"""

import json
import os
import subprocess 
import copy

#config_file = "configs/supervision/2021_05_21_pqsar_feature_maker.json" 
config_file = "configs/temp_psar.json"
config_file = "configs/temp.json"

#config_file = "configs/supervision/2021_05_13_pqsar_shell_scan.json"
args = json.load(open(config_file, "r"))

config_path = os.path.join("configs", "temp")
os.makedirs(config_path, exist_ok=True)

iterative_split = True
dataset_split = False

defaults = args['universal_args']
launcher_args= args['launcher_args']
gpu_default = defaults.get("gpu", False)

slurm_defaults = defaults.get("_slurm_args", [{}])
time_default = slurm_defaults[0].get("time", "0-06:00:00")
exp_name = launcher_args.get("experiment_name", "dense")


iterative_args = args.get('iterative_args', []) if iterative_split else [None]
dataset_args = args.get('universal_args', {}).get("_dataset") if dataset_split else [None]

ctr = 0
for dataset_arg in dataset_args: 
    for iterative_arg in iterative_args: 
        iterative_arg = copy.deepcopy(iterative_arg)

        # Replace with iterative arg and make sure it's only one
        new_args = copy.deepcopy(args)
        new_args['launcher_args']['use_slurm'] = False

        use_gpu = iterative_arg.get("gpu", gpu_default)[0] 

        if iterative_split: 
            new_args["iterative_args"] = [iterative_arg]

        if dataset_split: 
            new_args["_dataset"] = [dataset_arg]

        output_config_name = f"{exp_name}_temp_{ctr}"
        output_config_path = os.path.join(config_path, f"{output_config_name}.json")

        new_time = iterative_arg.get("_slurm_args", slurm_defaults)[0].get("time",
                                                                           time_default)
        with open(output_config_path, "w") as fp: 
            output_str = json.dumps(new_args, indent=2)
            fp.write(output_str)

        # Export a new json file that has only this dataset and export 
        sbatch_args = f"--output=logs/{output_config_name}_%j.log -J dense -t {new_time}"

        if use_gpu: 
            sbatch_args = f"{sbatch_args} --gres=gpu:volta:1"

        python_string = f"python run_scripts/run_combinations_slurm.py {output_config_path}"
        bash_string=f"sbatch --export=CMD=\"{python_string}\"  {sbatch_args} run_scripts/generic_slurm.sh"
        print(bash_string)
        ctr += 1
        # Don't call
        subprocess.call(bash_string, shell=True)

