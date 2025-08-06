import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from utils import (
    extract_consumer_name,
    load_and_clean_csv,
    filter_target_dates,
    make_pivot,
    is_complete_year_data,
    calc_hourly_stats,
    plot_hourly_stats,
)

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

for _, row in df_list.iterrows():
    file_name = row['ファイル名']
    consumer_name = extract_consumer_name(file_name)
    contract_type = row['契約電力']
    path = os.path.join(data_dir, file_name)

    df_raw = load_and_clean_csv(path)
    if df_raw is None:
        continue

    df_raw["計測日"] = pd.to_datetime(df_raw["計測日"], errors='coerce', format='%Y/%m/%d')
    df_raw = df_raw.dropna(subset=["計測日"])
    df_raw["年"] = df_raw["計測日"].dt.year
    df_raw["月日"] = df_raw["計測日"].dt.strftime('%m-%d')
    file_valid = False

    for year in sorted(df_raw["年"].unique()):
        target_start = pd.to_datetime(f"{year}-{target_start_md}")
        target_end = pd.to_datetime(f"{year}-{target_end_md}")
        target_dates = pd.date_range(start=target_start, end=target_end)

        df_period = is_complete_year_data(df_raw, target_dates, expected_rows)
        if df_period is None:
            continue  # データ不完全な年はスキップ

        pivot = make_pivot(df_period)

        # 時間表記を2桁に揃える（例：'1' → '01'）
        pivot.index = pivot.index.astype(str).str.extract(r'(\d{1,2})')[0].astype(int).astype(str).str.zfill(2)
        if pivot.shape[1] != target_days or pivot.isnull().values.any():
            continue  # データ不足

        consumer_profiles_by_contract[contract_type].append((pivot, consumer_name, year))
        file_valid = True

    if not file_valid:
        excluded_files.append(f"{file_name}（データ不足または対象期間なし）")

# Mixupによる合成（契約電力区分ごとに）
random.seed(42)
mixup_index = 1
original_index =1

plt.figure()
plt.subplots_adjust(left=0.1, right=0.98)

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
        plot_hourly_stats(x, y, yerr, linestyle='-')
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
        plot_hourly_stats(x, y, yerr, linestyle='--')
        plot_hourly_stats(x, y, yerr, linestyle='--')

        for h, m, s in zip(x, y, yerr):
            output_rows.append({'Consumer': consumer_name, 'Contract': contract_type, 'Hour': int(h), 'Mean': m, 'Std': s})

    plt.xlabel('Time')
    plt.xlim(1, 24)
    plt.xticks(range(1, 25))
    plt.ylabel('Power Consumption [kWh]')
    plt.ylim(-1, 800)
    plt.grid(True)
    plt.savefig(f"output/power_consumption_hourly_mixup_restricted_{contract_type}.png")
    plt.close()
            

# 結果出力
valid_file_count = len(set(row['Consumer'] for row in output_rows if not str(row['Consumer']).startswith('Mixup')))
synthetic_count = len(set(row['Consumer'] for row in output_rows if str(row['Consumer']).startswith('Mixup')))
print('有効ファイル数:', valid_file_count)
print('合成された需要家数:', synthetic_count)
print('合計需要家数:', valid_file_count + synthetic_count)
print('除外されたファイルの数:', len(excluded_files))

# CSV出力
output_df = pd.DataFrame(output_rows)
os.makedirs('output', exist_ok=True)
csv_path = 'output/power_consumption_hourly_mixup_restricted.csv'
output_df.to_csv(csv_path, index=False)


# 全データをプロット
plt.figure()
plt.subplots_adjust(left=0.1, right=0.97)
for consumer, group in output_df.groupby('Consumer'):
    x = group['Hour'].values
    y = group['Mean'].values
    yerr = group['Std'].values
    linestyle = '--' if consumer.startswith('Mixup') else '-'
    plot_hourly_stats(x, y, yerr, linestyle=linestyle)
plt.xlabel('Time')
plt.xlim(1, 24)
plt.xticks(range(1, 25))
plt.ylabel('Power Consumption [kWh]')
plt.ylim(-1, 800)
plt.title('Original(solid) + Mixup(broken)')
plt.grid(True)
os.makedirs('output', exist_ok=True)
plt.savefig("output/power_consumption_hourly_mixup_restricted.png")

# オリジナルデータをプロット
df_original = output_df[~output_df['Consumer'].str.startswith('Mixup')]
plt.figure()
plt.subplots_adjust(left=0.1, right=0.97)
for consumer, group in df_original.groupby('Consumer'):
    x = group['Hour'].values
    y = group['Mean'].values
    yerr = group['Std'].values
    plot_hourly_stats(x, y, yerr, linestyle='-')
plt.xlabel('Time')
plt.xlim(1, 24)
plt.xticks(range(1, 25))
plt.ylabel('Power Consumption [kWh]')
plt.ylim(-1, 800)
plt.title('Original')
plt.grid(True)
os.makedirs('output', exist_ok=True)
plt.savefig("output/power_consumption_hourly_all.png")

# Mixupのみのデータを抽出してプロット
df_mixup = output_df[output_df['Consumer'].str.startswith('Mixup')]
plt.figure()
plt.subplots_adjust(left=0.1, right=0.97)
for consumer, group in df_mixup.groupby('Consumer'):
    x = group['Hour'].values
    y = group['Mean'].values
    yerr = group['Std'].values
    plot_hourly_stats(x, y, yerr, linestyle='--')
plt.xlabel('Time')
plt.xlim(1, 24)
plt.xticks(range(1, 25))
plt.ylabel('Power Consumption [kWh]')
plt.ylim(-1, 800)
plt.title('Mixup Only')
plt.grid(True)
plt.savefig("output/power_consumption_hourly_mixup_restricted_only.png")
plt.show()
