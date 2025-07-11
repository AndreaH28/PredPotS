# PredPotS

PredPotS is a web application designed to predict one electron reduction potential in aqueous enviroment using machine learning models trained on chemical data.

## Web Application

The live web app is available here:  
[PredPotS Web Application](https://predpots.ttk.hu/)  

> The web app uses pretrained, saved ML models to provide fast and reliable predictions.

## This GitHub Repository

This repository contains resources related to the underlying data and ML models powering PredPotS, including:

- **`data/`**: Contains the datasets  
  - `Datasets/` with CSV files of molecular data  
  - `xyz_files/` with zipped XYZ geometry files  

- **`model_training/`**: Python code used for data training the machine learning models that are deployed in the backend of the web application.

## Environment Setup

We provide a conda environment file `environment.yml` to help recreate the exact Python environment used for training.

```bash
conda env create -f environment.yml
conda activate predpots_env

## Usage

To run the training code, use the following command:

```bash
python train_models.py <filename> -model <MODEL>


    <filename>: CSV file with the required headers (id, smiles, true)

    -model <MODEL>: Specify the model type to train (mandatory)

Optional arguments
Argument	Description
-p [PATIENCE]	Set patience for early stopping
-dr [DROPOUT]	Set dropout rate
-s [SEED]	Set random seed
-sp [SPLIT]	Set train/test split ratio
-max [NBEPOCH]	Set maximum number of epochs
-min [MINEPOCH]	Set minimum number of epochs
-pred, --predict	Run in prediction mode
-folw, --followtraining	Follow training progress
-estop, --early_stopping	Enable early stopping

Available models

    AttentiveFPModel

    GraphConvModel

    GCNModel

    GATModel

    DAGModel

Example

python train_models.py RP_CheMBL_SMILES.csv -model GraphConvModel


