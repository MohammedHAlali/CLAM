from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from sklearn import metrics
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, help='train, val, test, or all', default='test')
parser.add_argument('--task', type=str)
parser.add_argument('--phase', type=str)
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))
print('models_dir = ', args.models_dir)

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir
print('splits_dir = ', args.splits_dir)

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_binary':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'datasets/binary_{}.csv'.format(args.phase),
                            data_dir= os.path.join(args.data_root_dir, '{}_binary'.format(args.phase)),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_multi':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'datasets/multi_{}.csv'.format(args.phase),
                            data_dir= os.path.join(args.data_root_dir, '{}_multi'.format(args.phase)),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

# elif args.task == 'tcga_kidney_cv':
#     args.n_classes=3
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_kidney_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tcga_kidney_20x_features'),
#                             shuffle = False, 
#                             print_info = True,
#                             label_dict = {'TCGA-KICH':0, 'TCGA-KIRC':1, 'TCGA-KIRP':2},
#                             patient_strat= False,
#                             ignore=['TCGA-SARC'])

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}
print('checkpoints path: ', ckpt_paths)

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    all_confusion_matrices = []
    all_labels = []
    all_preds = []
    for ckpt_idx in range(len(ckpt_paths)):
        print('=============================== checkpoint [{}/{}] ==============='.format(ckpt_idx, len(ckpt_paths)))
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        model, patient_results, test_error, auc, df, cm, results_dict  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        all_confusion_matrices.append(cm)
        #all_labels.append(results_dict['Y'])
        #all_preds.append(results_dict['Y_hat'])
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)
    #print('all confusion matrices: ', all_confusion_matrices)
    #avg_all_labels = np.mean(all_labels, axis=0)
    #avg_all_preds = np.mean(all_preds, axis=0)
    #print('avg all labels = ', avg_all_labels)
    #print('avg all preds = ', avg_all_preds)
    #print('confusion matrix from avg labels and preds = \n', metrics.confusion_matrix(y_true=avg_all_labels, y_pred=avg_all_preds))
    avg_cm = np.mean(all_confusion_matrices, axis=0)
    print('average of all confusion matrices: \n', avg_cm)
    print('average of all_auc: ', np.mean(all_auc))
    print('average of all_acc: ', np.mean(all_acc))
    #tn, fp, fn, tp = avg_confusion_matrix.ravel()
    #print('tn={}, fp={}, fn={}, tp={}'.format(tn, fp, fn, tp))
    if(avg_cm.shape[0] > 1):
        print('multi class classification')
        tp_0 = avg_cm[0, 0]
        tn_0 = avg_cm[1,1]+avg_cm[2,2]+avg_cm[2,1]+avg_cm[1,2]
        fp_0 = avg_cm[1,0]+avg_cm[2,0]
        fn_0 = avg_cm[0,1]+avg_cm[0,2]
        prec_0 = tp_0 / (tp_0 + fp_0)
        recall_0 = tp_0 / (tp_0 + fn_0)
        f1_0 = 2 * (prec_0 * recall_0) / (prec_0 + recall_0)
        print('for class 0: tp = {}, tn = {}, fp = {}, fn = {}'.format(tp_0, tn_0, fp_0, fn_0))
        print('precision: ', prec_0, ' recall: ', recall_0, 'f1: ', f1_0)
        tp_1 = avg_cm[1, 1]
        tn_1 = avg_cm[0,0]+avg_cm[2,2]+avg_cm[0,2]+avg_cm[2,0]
        fp_1 = avg_cm[2,1]+avg_cm[0,1]
        fn_1 = avg_cm[1,0]+avg_cm[1,2]
        prec_1 = tp_1 / (tp_1 + fp_1)
        recall_1 = tp_1 / (tp_1+fn_1)
        f1_1 = 2 * (prec_1 * recall_1) / (prec_1 + recall_1)
        print('for class 1: tp = {}, tn = {}, fp = {}, fn = {}'.format(tp_1, tn_1, fp_1, fn_1))
        print('precision: ', prec_1, ' recall: ', recall_1, ' f1: ', f1_1)
        tp_2 = avg_cm[2,2]
        tn_2 = avg_cm[0,0]+avg_cm[1,1]+avg_cm[0,1]+avg_cm[1,0]
        fp_2 = avg_cm[0,2]+avg_cm[1,2]
        fn_2 = avg_cm[2,0]+avg_cm[2,1]
        prec_2 = tp_2 / (tp_2 + fp_2)
        recall_2 = tp_2 / (tp_2 + fn_2)
        f1_2 = 2 * (prec_2 * recall_2) / (prec_2 + recall_2)
        print('for class 2: tp = {}, tn = {}, fp = {}, fn = {}'.format(tp_2, tn_2, fp_2, fn_2))
        print('precision: ', prec_2, ' recall: ', recall_2, ' f1: ', f1_2)
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
