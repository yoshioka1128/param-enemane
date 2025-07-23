import pandas as pd
import matplotlib.pyplot as plt
import os

# データリスト読み込み
df_list = pd.read_csv('OPEN_DATA_60/list_60.csv', encoding='cp932')
df_list.columns = df_list.columns.str.strip()

data_dir = 'OPEN_DATA_60/raw'
plt.figure()
output_rows = []
excluded_files = []

# 対象とする月日（4月1日から5月31日）
target_start_md = '04-01'
target_end_md = '05-31'
target_days = 61
hours_per_day = 24
expected_rows = target_days * hours_per_day

for idx, row in df_list.iterrows():
    file_name = row['ファイル名']
    consumer_name = file_name.replace('.csv', '')
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
            # その年の04-01〜05-31の日付リストを作成
            target_start = pd.to_datetime(f"{year}-{target_start_md}")
            target_end = pd.to_datetime(f"{year}-{target_end_md}")
            target_dates = pd.date_range(start=target_start, end=target_end)

            df_period = df_raw[df_raw["計測日"].isin(target_dates)]

            if len(df_period) != expected_rows:
                continue  # この年のデータが不完全

            # ピボットして平均と標準偏差を計算
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

            file_valid = True  # 少なくとも1年が有効ならOK
            # 1年分だけで十分なら break

        if not file_valid:
            excluded_files.append(f"{file_name}（データ不足または対象期間なし）")

    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        continue

# 結果出力
valid_file_count = len(set([row['Consumer'] + str(row['Year']) for row in output_rows]))
print('有効ファイル数（年ごとの組み合わせ）:', valid_file_count)
print('\n除外されたファイル（期間不一致やデータ不足）:')
for f in excluded_files:
    print(f)
print(f'\n除外されたファイルの数:', len(excluded_files))

# グラフ保存
plt.xlabel('Hour of Day')
plt.ylabel('Predicted Negawatt [kWh]')
plt.title('Predicted Negawatt (Apr 1 - May 31, all years)')
plt.grid(True)
os.makedirs('output', exist_ok=True)
plt.savefig("output/predicted_negawatt_apr_may_all_years.png")
plt.show()
plt.close()

# CSV出力
output_df = pd.DataFrame(output_rows)
output_df.to_csv('output/predicted_negawatt_hourly_stats_all_years.csv', index=False)
