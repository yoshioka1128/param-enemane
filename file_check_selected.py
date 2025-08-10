import pandas as pd
import os
import random
import numpy as np
from utils import process_files, calc_hourly_stats

# データリスト読み込み
df_list = pd.read_csv('OPEN_DATA_60/list_60.csv', encoding='cp932')
df_list.columns = df_list.columns.str.strip()

data_dir = 'OPEN_DATA_60/raw'

output_rows = []
excluded_files = []

# 対象とする月日（4月1日から5月31日）
target_start_md = '04-01'
target_end_md = '05-31'
target_days = 61
hours_per_day = 24
expected_rows = target_days * hours_per_day

# 契約電力区分ごとのプロファイル格納用辞書
consumer_profiles_by_contract = {'低圧': [], '高圧小口': [], '高圧': []}


# --- データ読み込み ---
process_files(
    df_list, data_dir, target_start_md, target_end_md,
    expected_rows, target_days,
    consumer_profiles_by_contract,
    excluded_files
)

# Mixupによる合成（契約電力区分ごとに）
random.seed(42)
mixup_index = 1
original_index =1

for contract_type, profiles in consumer_profiles_by_contract.items():
    num_original = len(profiles)
    num_synthetic = int(num_original * 2.4)
    print(f"[{contract_type}] original: {num_original}, mixup: {num_synthetic}")

    # 元データをまず追加
    for p in profiles:
        x, y, yerr = calc_hourly_stats(p[0])
        consumer_name = f"Original{original_index}_{p[1]}"
        original_index += 1
        for h, m, s in zip(x, y, yerr):
            output_rows.append({'Consumer': consumer_name, 'Contract': contract_type, 'Hour': int(h), 'Mean': m, 'Std': s})

    for i in range(num_synthetic):
        if len(profiles) < 2:
            continue
        a, b = random.sample(profiles, 2)
        pivot_a = a[0].copy()
        pivot_b = b[0].copy()
        lam = random.uniform(0.3, 0.7)
        pivot_mix_values = lam * pivot_a.values + (1 - lam) * pivot_b.values
        pivot_mix = pd.DataFrame(pivot_mix_values, index=pivot_a.index)

        x, y, yerr = calc_hourly_stats(pivot_mix)
        
        consumer_name = f"Mixup{mixup_index}_{a[1]}_{b[1]}_lam={lam:.2f}"
        mixup_index += 1

        for h, m, s in zip(x, y, yerr):
            output_rows.append({'Consumer': consumer_name, 'Contract': contract_type, 'Hour': int(h), 'Mean': m, 'Std': s})

# 結果出力
valid_file_count = len(set(row['Consumer'] for row in output_rows if not str(row['Consumer']).startswith('Mixup')))
synthetic_count = len(set(row['Consumer'] for row in output_rows if str(row['Consumer']).startswith('Mixup')))
print('有効ファイル数:', valid_file_count)
print('合成された需要家数:', synthetic_count)
print('合計需要家数:', valid_file_count + synthetic_count)
print('除外されたファイルの数:', len(excluded_files))
