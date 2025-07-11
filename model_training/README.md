# PredPotS

**PredPotS** is a web application designed to predict one electron reduction potential in aqueous enviroment using machine learning models trained on chemical data.

The live web app is available here: [PredPotS Web Application](https://predpots.ttk.hu/)  

## Model Training Contents

This folder contains the script to train the models, the environment setup, and example data.

- **`train_models.py`** — The main training script to run different machine learning models on your dataset.  
- **`environment.yml`** — Conda environment file to recreate the exact software setup.  
- **`RP_CheMBL_SMILES.csv`** — Example dataset showing the required CSV format for training.

This setup ensures reproducibility of the training process used in **PredPotS**.

## Environment Setup

We provide a conda environment file `environment.yml` to help recreate the exact Python environment used for training.

```bash
conda env create -f environment.yml
conda activate predpots_env
```

## Usage

To run the training code, use the following command:

```bash
python train_models.py <filename> -model <MODEL>
```

    <filename>: CSV file with the required headers (id, smiles, true)

    -model <MODEL>: Specify the model type to train (mandatory)

### Available models
    AttentiveFPModel
    GraphConvModel
    GCNModel
    GATModel
    DAGModel

### Optional Arguments

| Argument               | Description                               | Default Value |
|------------------------|-------------------------------------------|---------------|
| `-p [PATIENCE]`        | Set patience for early stopping           | 2             |
| `-dr [DROPOUT]`        | Set dropout rate                          | 0             |
| `-s [SEED]`            | Set random seed                          | 289           |
| `-sp [SPLIT]`          | Set train/test split ratio                | 42            |
| `-max [NBEPOCH]`       | Set maximum number of epochs               | 150           |
| `-min [MINEPOCH]`      | Set minimum number of epochs               | 100           |
| `-pred, --predict`     | Run in prediction mode                     | False         |
| `-folw, --followtraining` | Follow training progress                | False         |
| `-estop, --early_stopping` | Enable early stopping                  | True          |

**Note:** If these optional arguments are not provided, the script will use default values defined in the code.

### Example

```bash
python train_models.py RP_CheMBL_SMILES.csv -model GraphConvModel
```

### Output Files and Directories

After running the training script, you will find several output files and folders generated:

- **`best_models/`**  
  Contains the best model saved according to early stopping criteria.

- **`statistics_<MODEL>_<epoch>.txt`**  
  Training and validation statistics logged at the specified best epoch.

- **`prediction_<MODEL>_<epoch>.txt`**  
  Model predictions on all data points at the best epoch. Each row includes a ‘Set’ column indicating the subset (train, valid, or test) for each entry.

- **`metric_MAE_<MODEL>_<epoch>.txt`**  
  Mean Absolute Error (MAE) per epoch for all subsets: train, valid, and test.

- **`metric_loss_<MODEL>_<epoch>.txt`**  
  Loss values recorded for each epoch, reported separately for train, valid, and test subsets.

- **`valid_save/`**  
- **`early_stopping_save/`**  
- **`<MODEL>_saved/`**  
  These directories contain intermediate saved files created during training. They are used for checkpointing and progress tracking but are not needed for final model evaluation.

**Note:** The `<epoch>` number corresponds to the training epoch where the model performed best and was saved.  
For detailed explanations please refer to the  [PredPotS Web Application](https://predpots.ttk.hu/) or the related [publication](YOUR_PAPER_LINK).


