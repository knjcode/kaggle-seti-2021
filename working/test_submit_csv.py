import pandas as pd
import torch
import numpy as np

from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--oof', nargs='+')
    parser.add_argument('-s', '--save_csv', type=str, default=None)
    parser.add_argument('-c', '--target_column', type=str, default='target')
    parser.add_argument('--ensemble_sigmoid', action='store_true')
    parser.add_argument('--power_ensemble', type=float, default=None)

    args = parser.parse_args()

    df = None
    for oof_csv in args.oof:
        cur_df = pd.read_csv(oof_csv)

        if args.ensemble_sigmoid:
            #cur_df['target'] = torch.tensor(cur_df['target'].values).sigmoid().numpy()
            cur_df[args.target_column] = torch.tensor(cur_df[args.target_column].values).sigmoid().numpy()
        else:
            cur_df[args.target_column] = cur_df[args.target_column] / 5.0

        if args.power_ensemble:
            cur_df[args.target_column] = np.power(cur_df[args.target_column], args.power_ensemble)

        if df is not None:
            df['target'] = df['target'] + cur_df[args.target_column]
        else:
            if args.target_column != 'target':
                cur_df = cur_df.drop('target', axis=1)
                cur_df.columns = ['id', 'target']
            df = cur_df

    df['target'] = df['target'] / len(args.oof)

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)

