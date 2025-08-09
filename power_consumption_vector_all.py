import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import (
    load_and_clean_csv, extract_consumer_name,
    is_complete_year_data, calc_hourly_stats, plot_hourly_stats,
    make_pivot
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

for _, row in df_list.iterrows():
    file_name = row['ファイル名']
    consumer_name = extract_consumer_name(file_name)
    path = os.path.join(data_dir, file_name)

    df_raw = load_and_clean_csv(path)
    if df_raw is None:
        continue

    df_raw["年"] = df_raw["計測日"].dt.year
    df_raw["月日"] = df_raw["計測日"].dt.strftime('%m-%d')
    unique_years = df_raw["年"].unique()
    file_valid = False

    for year in sorted(unique_years):
        target_start = pd.to_datetime(f"{year}-{target_start_md}")
        target_end = pd.to_datetime(f"{year}-{target_end_md}")
        target_dates = pd.date_range(start=target_start, end=target_end)

        df_period = is_complete_year_data(df_raw, target_dates, expected_rows)
        if df_period is None:
            continue

        pivot = make_pivot(df_period)
        pivot.index = (
            pivot.index.astype(str)
            .str.extract(r'(\d{1,2})')[0]
            .astype(int)
            .astype(str)
            .str.zfill(2)
        )

        if pivot.shape[1] != target_days or pivot.isnull().values.any():
            continue
        x, y, yerr = calc_hourly_stats(pivot)
        plot_hourly_stats(x, y, yerr, linestyle="-")

        for h, m, s in zip(x, y, yerr):
            output_rows.append({'Consumer': consumer_name, 'Year': year, 'Hour': int(h), 'Mean': m, 'Std': s})

        file_valid = True

    if not file_valid:
        excluded_files.append(f"{file_name}（データ不足または対象期間なし）")

# 結果出力
valid_file_count = len(set([row['Consumer'] + str(row['Year']) for row in output_rows]))
print('有効ファイル数（年ごとの組み合わせ）:', valid_file_count)
print(f'除外されたファイルの数:', len(excluded_files))
print(excluded_files)

# グラフ保存
plt.xlabel('Time')
plt.xlim(1,24)
plt.xticks(range(1, 25))
plt.ylabel('Predicted Negawatt [kWh]')
plt.ylim(-1,800)
#plt.title('Predicted Negawatt (Apr 1 - May 31, all years)')
plt.grid(True)
os.makedirs('output', exist_ok=True)
plt.savefig("output/predicted_negawatt_apr_may_all_years.png")
plt.show()
plt.close()

# CSV出力
output_df = pd.DataFrame(output_rows)
output_df.to_csv('output/predicted_negawatt_hourly_stats_all_years.csv', index=False)
