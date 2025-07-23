import pandas as pd
import matplotlib.pyplot as plt
import os

# データリスト読み込み
df_list = pd.read_csv('OPEN_DATA_60/list_60.csv', encoding='cp932')
df_list.columns = df_list.columns.str.strip()

data_dir = 'OPEN_DATA_60/raw'
target_start = pd.to_datetime('2013-04-01')
target_end = pd.to_datetime('2013-05-31')
target_days = (target_end - target_start).days + 1
target_dates = pd.date_range(start=target_start, end=target_end)
target_date_strs = target_dates.strftime('%Y/%m/%d')

plt.figure()
output_rows = []
excluded_files = []

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
        # 計測日を日付型に変換
        df_raw["計測日"] = pd.to_datetime(df_raw["計測日"], errors='coerce', format='%Y/%m/%d')
        df_raw = df_raw.dropna(subset=["計測日"])

        min_date = df_raw["計測日"].min()
        max_date = df_raw["計測日"].max()

        # 対象期間すべてをカバーしていない場合は除外
        if min_date > target_start or max_date < target_end:
#            excluded_files.append(f"{file_name}（範囲外: {min_date.date()}～{max_date.date()}）")
            continue

        # 対象期間で再フィルタ
        df = df_raw[df_raw["計測日"].isin(target_dates)]

        # 61日×24時間 = 1464行あるか
        if len(df) != 61 * 24:
            excluded_files.append(f"{file_name}（データ不足: {len(df)} 行）")
            continue

        pivot = df.pivot(index='計測時間', columns='計測日', values='全体')
        hourly_mean = pivot.mean(axis=1)
        hourly_std = pivot.std(axis=1)

        x = hourly_mean.index.astype(int).values
        y = hourly_mean.values
        yerr = hourly_std.values

        plt.plot(x, y, label=consumer_name)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.1)


        # --- 12,13,14時を取り出して異常検出 ---
        try:
            y12 = hourly_mean.loc['12']
            y13 = hourly_mean.loc['13']
            y14 = hourly_mean.loc['14']
            threshold = 100.0
            print(y12, y13, y14)
            if y13 < (y12 + y14) / 2 - threshold:
                print(f"🔻 異常減少検出: {consumer_name} (13時だけ低下)")
        except KeyError:
            # 12,13,14時のデータが欠損している場合を無視
            pass

    
        for h, m, s in zip(x, y, yerr):
            output_rows.append({'Consumer': consumer_name, 'Hour': int(h), 'Mean': m, 'Std': s})

    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        continue

# 結果出力
print('有効ファイル数:', int(len(output_rows) / 24))
print('\n除外されたファイル（期間不一致やデータ不足）:')
for f in excluded_files:
    print(f)

# グラフ保存
plt.xlabel('Hour of Day')
plt.ylabel('Predicted Negawatt [kWh]')
plt.title('Predicted Negawatt on June 1')
plt.grid(True)
plt.savefig("output/predicted_june1_2023.png")
plt.show()
plt.close()

# CSV出力
output_df = pd.DataFrame(output_rows)
os.makedirs('output', exist_ok=True)
output_df.to_csv('output/predicted_negawatt_hourly_stats_2023.csv', index=False)

