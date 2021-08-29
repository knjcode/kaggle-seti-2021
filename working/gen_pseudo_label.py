import sys
import pandas as pd

target_csv = sys.argv[1]
save_file_name = sys.argv[2]

print("target_csv:", target_csv)
print("save_file_name:", save_file_name)

df = pd.read_csv(target_csv) 

sorted_df = df.sort_values('target').reset_index(drop=True)
sorted_df['target'] = sorted_df['target'] / 5.0

negative_target = sorted_df[:5000]
positive_target = sorted_df[-5000:]

pseudo_df = pd.concat([negative_target, positive_target]).reset_index(drop=True)

pseudo_df.to_csv(save_file_name, index=False)
print("saved:", save_file_name)

