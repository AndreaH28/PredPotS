import os
os.environ["TF_USE_LEGACY_KERAS"]="1"

import sys
import argparse
import textwrap
import re
from typing import Dict
from datetime import datetime
import warnings

import pandas as pd
import numpy as np

from rdkit import Chem
import deepchem as dc
from rdkit.Chem.MolStandardize import rdMolStandardize

import dgl
import dgllife
import tensorflow as tf
from tensorflow import keras
import tf_keras
import torch

import scipy.stats as st


"""### Read input:
*   read and standardize SMILES
*   prepare input dataset
"""

def get_input_simple(csv):
    smiles, true_vals, ids = [], [], []
    wrong_ids = []

    with open(csv) as infile:
        # Read and normalize header
        header = infile.readline().strip().split(',')
        header = [col.strip().lower() for col in header]

        # Required columns (case-insensitive)
        try:
            id_idx = header.index("id")
            smiles_idx = header.index("smiles")
            true_idx = header.index("true")
        except ValueError as e:
            raise ValueError("CSV must contain headers: id, smiles, true (case-insensitive)") from e

        for line in infile:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            line_list = line.split(',')
            if len(line_list) <= max(id_idx, smiles_idx, true_idx):
                continue  # skip incomplete rows

            try:
                mol_id = line_list[id_idx].strip()
                smi = line_list[smiles_idx].strip()
                true_val = line_list[true_idx].strip()

                # Standardize and check molecule validity
                standardized = rdMolStandardize.StandardizeSmiles(smi)
                mol = Chem.MolFromSmiles(standardized)

                _ = list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))

                # Store processed values
                smiles.append(standardized)
                true_vals.append(float(true_val))
                ids.append(mol_id)

            except Exception:
                wrong_ids.append(mol_id if 'mol_id' in locals() else "unknown")

    weights = [1.01] * len(smiles)
    frame = pd.DataFrame({
        "Ids": ids,
        "Smiles": smiles,
        "Pot": true_vals,
        "Weight": weights
    })

    return frame, ids

def get_smiles_for_prediction(csv):
    smiles, pot, ids = [], [], []
    wrong_ids = []
    in_codes = []
    with open(csv) as infile:
        for line in infile:
            line_list = line.strip().split()
            mol = Chem.MolFromSmiles(line_list[1])
            if line_list[1] not in CODES:
                try:
                    list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
                    smiles.append(line_list[1])
                    ids.append(line_list[0])
                except:
                    wrong_ids.append(line_list[0])
            else:
                in_codes.append(line_list[1])
    weights = [1.01 for i in range(len(smiles))]
    pot = [None for i in range(len(smiles))]
    frame = []
    frame.append(pd.DataFrame(data={"Ids": ids, "Smiles": smiles, "Pot": pot, "Weight": weights}))
    return pd.concat(frame, ignore_index=True)

def create_char_to_idx(filename: str, max_len: int = 250, smiles_field: str = "smiles") -> Dict[str, int]:
    smiles_df = pd.read_csv(filename)
    char_set = set()
    for smile in smiles_df[smiles_field]:
        if len(smile) <= max_len:
            char_set.update(set(smile))

    unique_char_list = list(char_set)
    unique_char_list += ["<pad>", "<unk>"]
    char_to_idx = {letter: idx for idx, letter in enumerate(unique_char_list)}
    return char_to_idx

"""### DEF Run the models:

*   perform training
*   perform validation
*   perform predictions
"""
def run_model(df, model_used, nb_epoch, follow_training=True, early_stopping=False,
              patience=3, predict=False, df_test='', min_epoch_num=0, dropout=0.0, seed=123, split_seed=42):
    warnings.filterwarnings("ignore")
    # Dealing with possible input errors
    if predict and type(df_test) == str:
        print('Error: If predict mode is on (predict=True is given) you should provide another dataframe ' +
              'for testing too (df_test=...). The code execution will stop here.')
        return
    if (not predict) and type(df_test) != str:
        print('Warning: if predict mode is off (predict=False is given), the dataframe given for testing ' +
              'will not be used.')
        proceed = input('Do you still wish to proceed? [y/[n]]')
        if not re.match(r'[Yy]', proceed):
            return

            # json from dataframes
    t_start = datetime.now()
    json = df.to_json(orient='records', lines=True)
    if predict:
        json2 = df_test.to_json(orient='records', lines=True)

    # Model availability
    models = ['ChemCeption',
              'GraphConvModel',
              'DAGModel',
              'TextCNNModel',
              'Smiles2Vec',
              'GATModel',
              'GCNModel',
              'AttentiveFPModel']
    if model_used not in models:
        print('Unknown model. See --help for the available models.')
        return

    # Featurizer
    featurizer = None
    if model_used == "ChemCeption":
        featurizer = dc.feat.SmilesToImage()
    if model_used in ("GraphConvModel", "DAGModel"):
        featurizer = dc.feat.ConvMolFeaturizer()
    if model_used == "TextCNNModel":
        featurizer = dc.feat.RawFeaturizer(smiles=True)
    if model_used == "Smiles2Vec":
        df.to_csv('Data.csv', index=False)
        char_dict = create_char_to_idx(filename='Data.csv', smiles_field='Smiles')
        featurizer = dc.feat.SmilesToSeq(char_dict)
    if model_used in ("GATModel", "GCNModel", "AttentiveFPModel"):
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    # Data formatting
    loader = dc.data.JsonLoader(tasks=["Pot"], feature_field="Smiles", featurizer=featurizer,
                                weight_field=["Weight"], id_field="Ids")
    dataset = loader.create_dataset(json)
    if predict:
        loader2 = dc.data.JsonLoader(tasks=["Pot"], feature_field="Smiles", featurizer=featurizer,
                                     weight_field=["Weight"], id_field="Ids")
        test_dataset = loader2.create_dataset(json2)

    # Data transforming
    if model_used == "DAGModel":
        trans = dc.trans.DAGTransformer()
        dataset = trans.transform(dataset)
        if predict:
            test_dataset = trans.transform(test_dataset)

    # Data splitting
    splitter = dc.splits.RandomSplitter()

#   original split seed for all calculations:
#    split_seed = 42

    if not predict:
        train, valid, test = splitter.train_valid_test_split(dataset=dataset, seed=split_seed,
                                                             frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        flags = []
        for i in dataset.ids:
            if i in train.ids:
                flags.append('train')
            elif i in valid.ids:
                flags.append('valid')
            else:
                flags.append('test')
    else:
        train, valid = splitter.train_test_split(dataset=dataset, seed=13, frac_train=0.9, frac_test=0.1)
        flags = []
        for i in dataset.ids:
            if i in train.ids:
                flags.append('train')
            else:
                flags.append('valid')
        test = test_dataset

    # Training and testing
    true_vals = [i[0] for i in dataset.y]
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
    #callback = dc.models.ValidationCallback(valid, int(len(dataset.X) * 0.1), [metric])

    #for reproducibility
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_used == "GraphConvModel":
    # best dropout = 0.1
        model = dc.models.GraphConvModel(n_tasks=1, mode='regression', dropout=dropout,
                                         model_dir="GraphConvModel_saved")
    if model_used == "DAGModel":
    # best dropout = 0.1
        model = dc.models.DAGModel(n_tasks=1, mode='regression', dropout=dropout,
                                   model_dir="DAG_saved")
    if model_used == "GATModel":
    # best dropout = 0.1
        model = dc.models.GATModel(n_tasks=1, mode='regression', dropout=dropout,
                                   model_dir="GAT_saved")
    if model_used == "GCNModel":
    # best dropout = 0.2 or 0.1
        model = dc.models.GCNModel(n_tasks=1, mode='regression', dropout=dropout,
                                   model_dir="GCN_saved")
    if model_used == "AttentiveFPModel":
        model = dc.models.AttentiveFPModel(n_tasks=1, mode='regression', dropout=dropout,
                                           model_dir="AttentiveFP_saved")

    if follow_training:
        fit_t0 = datetime.now()
        epochs_no_improve = 0
        min_val_loss = 1000
        train_loss, valid_loss = [], []
        train_mae, valid_mae, test_mae = [], [], []
        train_eval, valid_eval = [], []
        early_epoch_num = 0
        for i in range(nb_epoch):
            model.fit(train, nb_epoch=1, all_losses=train_loss)
            model.save_checkpoint(max_checkpoints_to_keep=patience + 1, model_dir="valid_save")
            train_mae.append(round(model.evaluate(train, [metric])['mean_absolute_error'], 4))
            valid_mae.append(round(model.evaluate(valid, [metric])['mean_absolute_error'], 4))
            test_mae.append(round(model.evaluate(test, [metric])['mean_absolute_error'], 4))
            train_eval.append(round(train_loss[-1], 4))
            valid_loss.append(round(model.evaluate(valid, [dc.metrics.Metric(dc.metrics.mean_squared_error)])['mean_squared_error'], 4))
            model.restore(model_dir="valid_save")
            print('Epoch %s' % (i + 1))
            print('Train MAE: %s  Valid MAE: %s  Test MAE: %s Train loss: %s  Valid loss: %s' % (
                train_mae[-1], valid_mae[-1], test_mae[-1], round(train_loss[-1], 4), round(valid_loss[-1], 4)))

            if valid_loss[-1] < min_val_loss:
                epochs_no_improve = 0
                early_epoch_num = i + 1
                min_val_loss = valid_loss[-1]
                model.save_checkpoint(max_checkpoints_to_keep=1, model_dir="early_stopping_save")
            elif valid_loss[-1] < valid_loss[-2]:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if early_stopping and (i > patience) and (epochs_no_improve == patience) and (i >= min_epoch_num):
                    model.restore(model_dir="early_stopping_save")
                    print('Number of min. epochs: %s' % (min_epoch_num))
                    print('Early stopping.\nNumber of epochs: %s' % (i + 1))
                    print("Model was restored to its best state: epoch", early_epoch_num)
                    nb_epoch = early_epoch_num
                    break
            if i == (nb_epoch - 1):
                model.restore(model_dir="early_stopping_save")
                nb_epoch = early_epoch_num
                print('Number of min. epochs: %s' % (min_epoch_num))
                print("Model was restored to its best state: epoch", early_epoch_num)
                if early_stopping:
                    print('Number of min. epochs: %s' % (min_epoch_num))
                    print('Early stopping did not occur.')
                else:
                    print("Early stopping was disabled.")

        with open('metric_loss_%s_%s.txt' % (model_used, nb_epoch), 'w') as f:
            f.write('\n'.join(['Epoch_num,Loss,Loss_type'] +
                              ['%s,%.4f,%s\n%s,%.4f,%s' % (n, tl, 'Train_loss', n, vl, 'Valid_loss') for (n, tl, vl) in
                               zip(range(1, len(train_eval) + 1), train_eval, valid_loss)]))
        with open('metric_MAE_%s_%s.txt' % (model_used, nb_epoch), 'w') as f:
            f.write('\n'.join(['Epoch_num,MAE,MAE_type'] +
                              ['%s,%.4f,%s\n%s,%.4f,%s\n%s,%.4f,%s' % (n, tl, 'Train_MAE', n, vl, 'Valid_MAE', n, tsl, 'Test_MAE')
                              for (n, tl, vl, tsl) in
                               zip(range(1, len(train_loss) + 1), train_mae, valid_mae, test_mae)]))
        print("Fitting took %s time. (follow_training=True)" % (datetime.now() - fit_t0))
    else:
        fit_t0 = datetime.now()
        model.fit(train, nb_epoch=nb_epoch)
        print("Fitting took %s time. (follow_training=False)" % (datetime.now() - fit_t0))

    pred_dataset = model.predict(dataset)

    df_ids = [i for i in df['Ids']]
    if not predict:
        with open('prediction_%s_%s.txt' % (model_used, nb_epoch), 'w') as predfile:
            predfile.write('\n'.join([','.join(['Ids', 'True vals', 'Pred vals', 'Set'])] +
                                     ['%s,%s,%s,%s' % (ids, t, str(i[0]), f) for (ids, t, i, f) in
                                      zip(df_ids, true_vals, pred_dataset, flags)]))
    else:
        df_test_ids = [i for i in df_test['Ids']]
        with open('prediction_%s_%s.txt' % (model_used, nb_epoch), 'w') as predfile:
            predfile.write('\n'.join(['%-25s %15s %15s %s' % ('Ids', 'True vals', 'Pred vals', 'Set')] +
                                     ['%-25s %15s %15s %s' % (ids, t, str(i[0]), f) for (ids, t, i, f) in
                                      zip(df_ids, true_vals, pred_dataset, flags)]))
        new_pred = model.predict(test)
        with open('test_%s_%s.txt' % (model_used, nb_epoch), 'w') as predfile:
            predfile.write('\n'.join(['%-25s %s' % (ids, str(i[0])) for (ids, i) in zip(df_test_ids, new_pred)]))

    t_end = datetime.now()
    runtime = t_end - t_start
    save_filename = 'statistics_%s_%s.txt' % (model_used, nb_epoch)
    print("Training set score:", str(round(model.evaluate(train, [metric])['mean_absolute_error'], 4)))
    print("Validation set score:", str(round(model.evaluate(valid, [metric])['mean_absolute_error'], 4)))
    save_lines = ["Training set score (MAE): ", str(round(model.evaluate(train, [metric])['mean_absolute_error'], 4)),
                  "\nValidation set score (MAE): ",
                  str(round(model.evaluate(valid, [metric])['mean_absolute_error'], 4))]
    if not predict:
        print("Test set score:", str(round(model.evaluate(test, [metric])['mean_absolute_error'], 4)))
        save_lines.append("\nTest set score (MAE): ")
        save_lines.append(str(round(model.evaluate(test, [metric])['mean_absolute_error'], 4)))
    save_lines.append("\nTotal runtime: %s\n" % runtime)
    save_lines.append("\nUsed data and parameters:\n" +
#                      "    CSV file: %s\n"%(CSV) +
#                      "    CSV file for testing: %s\n"%(TEST_CSV) +
                      "    Model: %s\n"%(model_used) +
                      "    Dropout: %s\n"%(dropout) +
                      "    Epoch number (max): %s\n"%(nb_epoch) +
                      "    Final epoch number: %s\n"%(early_epoch_num) +
                      "    Early stopping: %s\n"%(early_stopping) +
                      "    Min epoch number: %s\n"%(min_epoch_num) +
                      "    SPLIT_SEED: %s\n"%(split_seed) +
                      "    MODEL_SEED: %s\n"%(seed))
    if early_stopping and follow_training:
        save_lines.append("    Patience: %s\n"%(patience) +
                          "    Min epoch number: %s\n"%(min_epoch_num) +
                          "    SPLIT_SEED: %s\n"%(split_seed) +
                          "    SEED: %s\n"%(seed))


    with open(save_filename, "w") as f:
        f.write(''.join(save_lines))

    model.save_checkpoint(max_checkpoints_to_keep=1,
                          model_dir="best_models/%s_%s" % (model_used, nb_epoch))
    return

def main():
    parser = argparse.ArgumentParser(
            prog='train_models.py', 
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent('''\
                      additional information:
                      Provide the DATASET and MODEL for training!
                      MODELS available: 
                      AttentiveFPModel, GraphConvModel, GCNModel, GATModel, DAGModel
                      '''))

    parser.add_argument(dest='filename', metavar='filename')
    parser.add_argument('-p', '--patience', nargs='?', const=2, default=2, type=int)
    parser.add_argument('-dr', '--dropout', nargs='?', const=0.0, default=0.0, type=float)
    parser.add_argument('-s', '--seed', nargs='?', const=289,default=289, type=int)
    parser.add_argument('-sp', '--split', nargs='?', const=42,default=42, type=int)
    parser.add_argument('-max', '--nbepoch', nargs='?', const=150, default=150, type=int)
    parser.add_argument('-min', '--minepoch', nargs='?', const=100, default=100, type=int)
    parser.add_argument('-pred', '--predict', action='store_true')
    parser.add_argument('-folw', '--followtraining', action='store_true')
    parser.add_argument('-estop', '--early_stopping', action='store_false')
    parser.add_argument('-model', '--model', required=True)
    parser.print_help()
    args = parser.parse_args()
    options = parser.parse_args()

    if len(args.filename) == 0:
        parser.print_help()
        sys.exit('\n Please provide the dataset for trainng! \n')
    else: 
        CSV = args.filename

    PATIENCE = args.patience
    DROPOUT = args.dropout
    SEED = args.seed
    SPLIT = args.split
    NB_EPOCH = args.nbepoch
    MIN_EPOCHS = args.minepoch
    PREDICT = args.predict
    FOLLOW_TRAINING = args.followtraining
    EARLY_STOPPING = args.early_stopping 
    MODEL = args.model

    if EARLY_STOPPING:
        FOLLOW_TRAINING = True

# Temporarly this is not used at all:
    TEST_CSV = ''

# Default setup is to Train the models
# for training the models:
# PREDICT = False
# FOLLOW_TRAINING = True
# EARLY_STOPPING = True

    print(CSV, MODEL, SEED, NB_EPOCH)

    df, CODES = get_input_simple(CSV)

    if PREDICT:
        df_test = get_smiles_for_prediction(TEST_CSV)
        run_model(df, MODEL, nb_epoch=NB_EPOCH, predict=True, df_test=df_test,
              follow_training=FOLLOW_TRAINING, early_stopping=EARLY_STOPPING,
              patience=PATIENCE, 
              min_epoch_num=MIN_EPOCHS, dropout=DROPOUT, 
              seed=SEED, split_seed=SPLIT)
    else:
        run_model(df, MODEL, nb_epoch=NB_EPOCH, predict=False, follow_training=FOLLOW_TRAINING,
              early_stopping=EARLY_STOPPING, 
              patience=PATIENCE, 
              min_epoch_num=MIN_EPOCHS, dropout=DROPOUT,
              seed=SEED, split_seed=SPLIT)

if __name__== "__main__":
  main()


