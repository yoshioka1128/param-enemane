import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from utils import (
    load_and_clean_csv, extract_consumer_name,
    is_complete_year_data, calc_hourly_stats, plot_hourly_stats
)

# データリスト読み込み
df_list = pd.read_csv('OPEN_DATA_60/list_60.csv', encoding='cp932')
df_list.columns = df_list.columns.str.strip()

data_dir = 'OPEN_DATA_60/raw'
plt.figure()
plt.subplots_adjust(left=0.1, right=0.97)

output_rows = []
excluded_files = []

# 対象とする月日（4月1日から5月31日）
target_start_md = '04-01'
target_end_md = '05-31'
target_days = 61
hours_per_day = 24
expected_rows = target_days * hours_per_day

consumer_profiles = []

for _, row in df_list.iterrows():
    file_name = row['ファイル名']
    consumer_name = file_name.replace('.csv', '')
    path = os.path.join(data_dir, file_name)

    df_raw = load_and_clean_csv(path)
    if df_raw is None:
        continue

    df_raw["計測日"] = pd.to_datetime(df_raw["計測日"], errors='coerce', format='%Y/%m/%d')
    df_raw = df_raw.dropna(subset=["計測日"])
    df_raw["年"] = df_raw["計測日"].dt.year
    df_raw["月日"] = df_raw["計測日"].dt.strftime('%m-%d')
    unique_years = df_raw["年"].unique()
    file_valid = False

    for year in sorted(unique_years):
        target_start = pd.to_datetime(f"{year}-{target_start_md}")
        target_end = pd.to_datetime(f"{year}-{target_end_md}")
        target_dates = pd.date_range(start=target_start, end=target_end)

        df_period = df_raw[df_raw["計測日"].isin(target_dates)]

        if len(df_period) != expected_rows:
            continue

        pivot = make_pivot(df_period)        
        x, y, yerr = calc_hourly_stats(pivot)
        plot_hourly_stats(x, y, yerr, label=f"{consumer_name} ({year})")

        for h, m, s in zip(x, y, yerr):
            output_rows.append({'Consumer': consumer_name, 'Year': year, 'Hour': int(h), 'Mean': m, 'Std': s})

        consumer_profiles.append((x, y, yerr))
        file_valid = True

    if not file_valid:
        excluded_files.append(f"{file_name}（データ不足または対象期間なし）")

# Mixupによる合成
num_original = len(consumer_profiles)
num_synthetic = int(num_original * 2.5)

for i in range(num_synthetic):
    a, b = random.sample(consumer_profiles, 2)
    lam = random.uniform(0.3, 0.7)
    x = a[0]
    y_mix = lam * a[1] + (1 - lam) * b[1]
#    yerr_mix = lam * a[2] + (1 - lam) * b[2] # old
    yerr_mix = np.sqrt(lam * a[2]**2 + (1 - lam) * b[2]**2)
    label = f"Mixup_{i+1}"
    plt.plot(x, y_mix, label=label, linestyle='--')
    plt.fill_between(x, y_mix - yerr_mix, y_mix + yerr_mix, alpha=0.1)

    for h, m, s in zip(x, y_mix, yerr_mix):
        output_rows.append({'Consumer': label, 'Year': 'Synthetic', 'Hour': int(h), 'Mean': m, 'Std': s})

# 結果出力
valid_file_count = len(set([row['Consumer'] + str(row['Year']) for row in output_rows if row['Year'] != 'Synthetic']))
synthetic_count = len(set([row['Consumer'] for row in output_rows if row['Year'] == 'Synthetic']))
print('有効ファイル数（年ごとの組み合わせ）:', valid_file_count)
print('合成された需要家数:', synthetic_count)
print(f'\n除外されたファイルの数:', len(excluded_files))

# グラフ保存
plt.xlabel('Time')
plt.xlim(1,24)
plt.xticks(range(1, 25))
plt.ylabel('Predicted Negawatt [kWh]')
plt.ylim(-1,800)
plt.title('Original(solid) + Mixup(broken)')
plt.grid(True)
os.makedirs('output', exist_ok=True)
plt.savefig("output/predicted_negawatt_apr_may_mixup.png")
plt.show()
plt.close()

# Mixupのみのデータを抽出してプロット
df = pd.read_csv('output/predicted_negawatt_hourly_stats_mixup.csv')
df_mixup = df[df['Consumer'].str.startswith('Mixup_')]

plt.figure()
plt.subplots_adjust(left=0.1, right=0.97)
for (consumer, year), group in df_mixup.groupby(['Consumer', 'Year']):
    x = group['Hour'].values
    y = group['Mean'].values
    yerr = group['Std'].values
    plt.plot(x, y, linestyle='--', label=consumer)
    plt.fill_between(x, y - yerr, y + yerr, alpha=0.1)

plt.xlabel('Time')
plt.xlim(1,24)
plt.xticks(range(1, 25))
plt.ylabel('Predicted Negawatt [kWh]')
plt.ylim(-1,800)
plt.title('Mixup Only')
plt.grid(True)
os.makedirs('output', exist_ok=True)
plt.savefig("output/predicted_negawatt_apr_may_mixup_only.png")
plt.show()
plt.close()

# CSV出力
output_df = pd.DataFrame(output_rows)
output_df.to_csv('output/predicted_negawatt_hourly_stats_mixup.csv', index=False)

