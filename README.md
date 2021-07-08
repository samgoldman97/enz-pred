# Enzyme Promiscuity Prediction

This repository contains code used to compare various different enzyme-substrate promiscuity strategies on family-wide enzyme screening data. 


## Install

### Packages

A python environment can be created directly using the `environment.yml` file included:

`conda env create -f environment.yml`

Once an enviornment has been activated, the package can be installed with:

`python setup.py install`

### Featurizations 

All featurizations can be handeled directly by the `build_features` file. Due to the cost of repeatedly using language models to featurize proteins, features are automatically cached in `data/program_cache` for later use if the `--cache-dir` argument is set in the program. 

Precomputed JT-VAE embeddings were also precomputed using a forked [repository](https://github.com/samgoldman97/icml18-jtnn)from the original JT-VAE [paper](https://arxiv.org/abs/1802.04364). These can be found in `data/processed/precomputed_features/`. 

## Dataset

The datasets used in this study and the corresponding structure reference files can be downloaded from the following github repository: `https://github.com/samgoldman97/enzyme-datasets`, which contains instructions for how datasets, alignments, and structure references were created and processed. 

These dataset files are also included within this package directly for convenience in `data/processed/`. 

## Testing a model

### Launching a simple program

A simple, example program can be executed using the following run call:

```
python run_scripts/run_combinations_slurm.py configs/2021_06_30_example_launch.json
```

This will launch an evaluation of a KNN based model that uses the levenshtein distance between enzymes sequences to make predictions about held out enzyme activity for each substrate in the esterase_binary dataset. 

### Running experiments


Experiments can be run using `python train_model.py`. Experiments can also be run from config files located in configs using the launcher scripts contained in `run_scripts`. Specifically, `python run_scripts/run_combinations_slurm.py [config file]` will launch the expriments defined in the config file, with instructions for config files contained at the top of `run_combinations_slurm.py`. The config files have an optional flag to run the program on a SLURM cluster for parallelization as done in the original study. 

The various provided config files are detailed here: 

1. `configs/2021_05_25_psar_olea_hyperopt.json`: Perform hyperoptimization for various model types on the OleA dataset for PSAR models that try to generalize to new enzymes.  

2. `configs/2021_05_25_qsar_olea_hyperopt.json`: Perform hyperoptimization for various model types on the OleA dataset for QSAR models that try to generalize to new substrates.   

3. `configs/2021_05_27_psar_multi.json`: Use the resulting hyperoptimized parameters to run PSAR analyses on all other datasets.  

4. `configs/2021_05_28_qsar_multi.json`: Use the resulting hyperoptimized parameters to run QSAR analyses.

5. `configs/2021_05_25_psar_olea_hyperopt.json`: Use the resulting PSAR hyperoptimized parameters to run pooling comparison experiments in the PSAR direction.  

6. `configs/2021_06_30_example_launch.json`: Run an example program launch


After completing a set of experiments, all the results entries from the specific experiment can be collected into a single results file using the script `run_scripts/combine_csvs.py`. For instance, to combine any experiments in the example launch: 

```
python run_scripts/combine_csvs.py --results-dir results/dense/2021_06_30_example_launch -
-out-file results/dense/2021_06_30_example_launch/combined_csv.csv
```

These combined results files are used 

## Making figures

Figures can be constructed usign the scripts contained in the folder `make_figs/`. Assumign the proper folders. All figure scripts can be run using the command: 

```
source make_figs/make_all_figs.sh
```

TODO: 
- Add in download from already finished results to make all figures