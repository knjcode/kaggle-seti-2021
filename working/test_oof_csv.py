import pandas as pd
import torch
import numpy as np

from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--oof', nargs='+')
    parser.add_argument('-s', '--save_csv', type=str, default=None)
    parser.add_argument('--ensemble_sigmoid', action='store_true')
    parser.add_argument('--power_ensemble', type=float, default=None)

    args = parser.parse_args()

    df = None
    for oof_csv in args.oof:
        cur_df = pd.read_csv(oof_csv)

        if args.ensemble_sigmoid:
            cur_df['preds'] = torch.tensor(cur_df['preds'].values).sigmoid().numpy()

        if args.power_ensemble:
            cur_df['preds'] = np.power(cur_df['preds'], args.power_ensemble)

        if df is not None:
            df['preds'] = df['preds'] + cur_df['preds']
        else:
            df = cur_df

    df['preds'] = df['preds'] / len(args.oof)

    score = roc_auc_score(df.target, df.preds)
    print("oof score:", score)

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
