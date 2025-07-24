import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import utils

# データリスト読み込み
df_list = pd.read_csv('OPEN_DATA_60/list_60.csv', encoding='cp932')
df_list.columns = df_list.columns.str.strip()

data_dir = 'OPEN_DATA_60/raw'
plt.figure()
plt.subplots_adjust(left=0.1, right=0.98)

output_rows = []
excluded_files = []

# 対象とする月日（4月1日から5月31日）
target_start_md = '04-01'
target_end_md = '05-31'
target_days = 61
hours_per_day = 24
expected_rows = target_days * hours_per_day

# 契約電力区分ごとのプロファイル格納用辞書
consumer_profiles_by_contract = {'低圧': [], '高圧': [], '高圧小口': []}

for idx, row in df_list.iterrows():
    file_name = row['ファイル名']
    consumer_name = file_name.replace('.csv', '')
    contract_type = row['契約電力']
    path = os.path.join(data_dir, file_name)

    if not os.path.isfile(path):
        continue

    try:
        df_raw = pd.read_csv(
            path,
            encoding='utf-8-sig',
            usecols=[0, 1, 2],
            header=0,
            names=["計測日", "計測時間", "全体"]
        )

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

            pivot = df_period.pivot(index='計測時間', columns='計測日', values='全体')
            hourly_mean = pivot.mean(axis=1)
            hourly_std = pivot.std(axis=1)

            x = hourly_mean.index.astype(int).values
            y = hourly_mean.values
            yerr = hourly_std.values

            label = f"{consumer_name} ({year})"
            plt.plot(x, y, label=label)
            plt.fill_between(x, y - yerr, y + yerr, alpha=0.1)

            for h, m, s in zip(x, y, yerr):
                output_rows.append({'Consumer': consumer_name, 'Year': year, 'Hour': int(h), 'Mean': m, 'Std': s})

            consumer_profiles_by_contract[contract_type].append((x, y, yerr))
            file_valid = True

        if not file_valid:
            excluded_files.append(f"{file_name}（データ不足または対象期間なし）")

    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        continue

# Mixupによる合成（契約電力区分ごとに）
mixup_index = 1
for contract_type, profiles in consumer_profiles_by_contract.items():
    num_original = len(profiles)
    num_synthetic = int(num_original * 2.5)

    for i in range(num_synthetic):
        if len(profiles) < 2:
            continue
        a, b = random.sample(profiles, 2)
        lam = random.uniform(0.3, 0.7)
        x = a[0]
        y_mix = lam * a[1] + (1 - lam) * b[1]
        yerr_mix = np.sqrt(lam * a[2]**2 + (1 - lam) * b[2]**2)
        label = f"Mixup_{mixup_index} ({contract_type})"
        mixup_index += 1
        plt.plot(x, y_mix, label=label, linestyle='--')
        plt.fill_between(x, y_mix - yerr_mix, y_mix + yerr_mix, alpha=0.1)

        for h, m, s in zip(x, y_mix, yerr_mix):
            output_rows.append({'Consumer': label, 'Year': 'Synthetic', 'Hour': int(h), 'Mean': m, 'Std': s})

# 結果出力
valid_file_count = len(set([row['Consumer'] + str(row['Year']) for row in output_rows if row['Year'] != 'Synthetic']))
synthetic_count = len(set([row['Consumer'] for row in output_rows if row['Year'] == 'Synthetic']))
print('有効ファイル数（年ごとの組み合わせ）:', valid_file_count)
print('合成された需要家数:', synthetic_count)
print('\n除外されたファイル（期間不一致やデータ不足）:')
for f in excluded_files:
    print(f)
print(f'\n除外されたファイルの数:', len(excluded_files))

# グラフ保存
plt.xlabel('Time')
plt.xlim(1, 24)
plt.xticks(range(1, 25))
plt.ylabel('Predicted Negawatt [kWh]')
plt.ylim(-1, 800)
plt.title('Original(solid) + Mixup(broken)')
plt.grid(True)
os.makedirs('output', exist_ok=True)
plt.savefig("output/predicted_negawatt_apr_may_mixup_restricted.png")
plt.show()
plt.close()

# CSV出力
output_df = pd.DataFrame(output_rows)
os.makedirs('output', exist_ok=True)
csv_path = 'output/predicted_negawatt_hourly_stats_mixup_restricted.csv'
output_df.to_csv(csv_path, index=False)

# Mixupのみのデータを抽出してプロット
df_mixup = output_df[output_df['Consumer'].str.startswith('Mixup_')]

plt.figure()
plt.subplots_adjust(left=0.1, right=0.97)
for (consumer, year), group in df_mixup.groupby(['Consumer', 'Year']):
    x = group['Hour'].values
    y = group['Mean'].values
    yerr = group['Std'].values
    plt.plot(x, y, linestyle='--', label=consumer)
    plt.fill_between(x, y - yerr, y + yerr, alpha=0.1)

plt.xlabel('Time')
plt.xlim(1, 24)
plt.xticks(range(1, 25))
plt.ylabel('Predicted Negawatt [kWh]')
plt.ylim(-1, 800)
plt.title('Mixup Only')
plt.grid(True)
plt.savefig("output/predicted_negawatt_apr_may_mixup_restricted_only.png")
plt.show()
