import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from utils import process_files, calc_hourly_stats, plot_hourly_stats

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

# Original（契約電力区分ごとに）
random.seed(42)
original_index =1

for contract_type, profiles in consumer_profiles_by_contract.items():
    plt.figure()
    plt.subplots_adjust(left=0.1, right=0.98)

    num_original = len(profiles)
    num_synthetic = int(num_original * 2.4)

    # 元データをまず追加
    for p in profiles:
        x, y, yerr = calc_hourly_stats(p[0])
        consumer_name = f"Original{original_index}_{p[1]}"
        original_index += 1
        plot_hourly_stats(x, y, yerr, linestyle='-')
        for h, m, s in zip(x, y, yerr):
            output_rows.append({'Consumer': consumer_name, 'Contract': contract_type, 'Hour': int(h), 'Mean': m, 'Std': s})

    plt.xlabel('Time')
    plt.xlim(1, 24)
    plt.xticks(range(1, 25))
    plt.ylabel('Power Consumption [kWh]')
    plt.ylim(-1, 800)
    plt.grid(True)
    plt.savefig(f"output/power_consumption_hourly_original_{contract_type}.png")
    plt.close()
            

# 結果出力
valid_file_count = len(set(row['Consumer'] for row in output_rows if not str(row['Consumer']).startswith('Mixup')))
synthetic_count = len(set(row['Consumer'] for row in output_rows if str(row['Consumer']).startswith('Mixup')))
print('有効ファイル数（年ごとの組み合わせ）:', valid_file_count)
print('合成された需要家数:', synthetic_count)
print('合計需要家数:', valid_file_count + synthetic_count)
print('除外されたファイルの数:', len(excluded_files))

# CSV出力
output_df = pd.DataFrame(output_rows)
os.makedirs('output', exist_ok=True)
csv_path = 'output/power_consumption_hourly_original.csv'
output_df.to_csv(csv_path, index=False)


# 全データをプロット
plt.figure()
plt.subplots_adjust(left=0.1, right=0.97)
for consumer, group in output_df.groupby('Consumer'):
    x = group['Hour'].values
    y = group['Mean'].values
    yerr = group['Std'].values
    linestyle = '--' if consumer.startswith('Mixup_') else '-'
    plot_hourly_stats(x, y, yerr, linestyle=linestyle)
plt.xlabel('Time')
plt.xlim(1, 24)
plt.xticks(range(1, 25))
plt.ylabel('Power Consumption [kWh]')
plt.ylim(-1, 800)
plt.title('Original')
plt.grid(True)
os.makedirs('output', exist_ok=True)
plt.savefig("output/power_consumption_hourly_original.png")
plt.show()
