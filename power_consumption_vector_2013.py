import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import (
    load_and_clean_csv, is_valid_period, filter_target_dates,
    make_pivot, extract_consumer_name, calc_hourly_stats, plot_hourly_stats
)

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

# データ格納
output_rows = []
excluded_files = []

for _, row in df_list.iterrows():
    file_name = row['ファイル名']
    consumer_name = extract_consumer_name(file_name)
    path = os.path.join(data_dir, file_name)

    df_raw = load_and_clean_csv(path)
    if df_raw is None or not is_valid_period(df_raw, target_start, target_end):
        continue

    df = filter_target_dates(df_raw, target_dates)
    if df is None:
        excluded_files.append(f"{file_name}（データ不足）")
        continue

    pivot = make_pivot(df)
#    pivot.index = (
#        pivot.index.astype(str)
#        .str.extract(r'(\d{1,2})')[0]
#        .astype(int)
#        .astype(str)
#        .str.zfill(2)
#    )
#    if pivot.shape[1] != target_days or pivot.isnull().values.any():
#        continue

    x, y, yerr = calc_hourly_stats(pivot)
    plot_hourly_stats(x, y, yerr, linestyle="-")

    for h, m, s in zip(x, y, yerr):
        output_rows.append({'Consumer': consumer_name, 'Hour': int(h), 'Mean': m, 'Std': s})

# 結果出力
print('有効な消費者数:', int(len(output_rows) / 24))
print('\n除外されたファイル（期間不一致やデータ不足）:')
for f in excluded_files:
    print(f)

# グラフ保存
plt.xlabel('Time')
plt.xlim(1,24)
plt.ylabel('Predicted Negawatt [kWh]')
plt.ylim(-1,800)
plt.grid(True)
plt.savefig("output/power_consumption_hourly_2013.png")
plt.show()
plt.close()

# CSV出力
output_df = pd.DataFrame(output_rows)
os.makedirs('output', exist_ok=True)
output_df.to_csv('output/power_consumption_hourly_2013.csv', index=False)

