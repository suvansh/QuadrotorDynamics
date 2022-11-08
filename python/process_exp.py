import sys
import os
from os.path import join
import pandas as pd
import numpy as np


outfile = "merged.csv"
dt = 10  # ms

def process_exp(exp_dir):
    dfs, arrs = [], []
    for filename in os.listdir(exp_dir):
        if not filename.endswith(".csv") or filename == outfile: continue
        df = pd.read_csv(join(exp_dir, filename))
        arr = df.values
        dfs.append(df)
        arrs.append(arr)
    # sample_period = np.diff(arrs[0][:2,0])
    start = np.around(max(arr[:,0].min() for arr in arrs), decimals=-1).astype(int)
    end = np.around(min(arr[:,0].max() for arr in arrs), decimals=-1).astype(int)
    knots = list(range(start, end+1, dt))  # times to sample at: every 10 ms between the latest start time and earliest end time across the files
    big_df = pd.DataFrame(knots, columns=['time'])
    for df, arr in zip(dfs, arrs):
         for i, col in enumerate(df.columns):
            if col in big_df.columns[1:]:
                print(f"Warning: column {col} being overwritten")
            big_df[col] = np.interp(knots, arr[:,0], arr[:,i])
    # from table 1 psi_1: https://arxiv.org/pdf/2012.05457.pdf. see https://forum.bitcraze.io/viewtopic.php?t=4761 for explanation on normalization
    for i in range(1, 5):
        ph = big_df[f'm{i}'] / 65536
        vh = big_df['vbat'] / 4.2
        big_df[f't{i}'] = (11.09 - 39.08*ph - 9.53*vh + 20.57*ph**2 + 38.43*vh**2) * 0.009807  # 0.009807 is conversion from gram-force to newtons
    big_df['alpha_x'] = big_df.gx.diff() / dt
    big_df['alpha_y'] = big_df.gy.diff() / dt
    big_df['alpha_z'] = big_df.gz.diff() / dt
    big_df = big_df.iloc[1:].set_index('time')
    big_df.to_csv(join(exp_dir, outfile))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_exp.py <path_to_exp_dir>")
        exit()
    exp_dir = sys.argv[1]
    process_exp(exp_dir)
