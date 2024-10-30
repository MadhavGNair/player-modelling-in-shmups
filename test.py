import json
import pandas as pd


df = pd.read_csv('data/yohaan/player_logs_run_3_10_20_2024_13_58_42.csv')

fired_df = df.loc[df[' Event'] == 'Bullet Fired']
missed_df = df.loc[df[' Event'] == 'Bullet Missed']
hit_df = df.loc[df[' Event'] == 'Enemy hit']

print(f'missed = {len(missed_df)}, hit = {len(hit_df)}, fired {len(fired_df)}')
print(f'missed + hit = {len(missed_df) + len(hit_df)}')
