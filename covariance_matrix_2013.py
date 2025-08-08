import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import (
    load_and_clean_csv,
    is_valid_period,
    filter_target_dates,
    make_pivot,
    extract_consumer_name
)

# 入力（ターミナルから時間帯を指定）
user_input = input("共分散行列を計算したい時間帯（0〜23）を入力してください（未入力なら12）: ")
target_hour = user_input if user_input else "12"
target_hour = target_hour.zfill(2)

# データ読み込み設定
list_path = 'OPEN_DATA_60/list_60.csv'
data_dir = 'OPEN_DATA_60/raw'
target_start = pd.to_datetime('2013-04-01')
target_end = pd.to_datetime('2013-05-31')
target_dates = pd.date_range(start=target_start, end=target_end)

# データ格納
data_matrix = []
consumer_names = []
excluded_files = []

# ファイルリスト読み込み
df_list = pd.read_csv(list_path, encoding='cp932')
df_list.columns = df_list.columns.str.strip()

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
    pivot.index = pivot.index.astype(str).str.extract(r'(\d{1,2})')[0].astype(int).astype(str).str.zfill(2)

    if target_hour not in pivot.index or pivot.shape[1] != 61:
        continue

    data_matrix.append(pivot.loc[target_hour].values)
    consumer_names.append(consumer_name)

# 結果出力
print(f"有効な消費者数（{target_hour}時）: {len(data_matrix)}")
print('\n除外されたファイル（期間不一致やデータ不足）:')
for f in excluded_files:
    print(f)

# 共分散行列の計算と表示
if len(data_matrix) > 1:
    cov_matrix = np.cov(np.array(data_matrix), ddof=True) # divided by n
    cov_df = pd.DataFrame(cov_matrix, index=consumer_names, columns=consumer_names)
    
    # CSVとして保存
    cov_df.to_csv(f"output/covariance_matrix_time{target_hour}_2023.csv", encoding='utf-8-sig')
    print(f"共分散行列を 'covariance_matrix_time{target_hour}_2023.csv' に保存しました。e")

    # ヒートマップとして保存
    plt.figure()
    sns.heatmap(cov_df, annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False, vmin=-25, vmax=25,
                cbar_kws={'label': 'Covariance'})
    plt.xlabel("Consumer")
    plt.ylabel("Consumer")
    plt.title(f"Covariance Matrix - Time {target_hour}")
    plt.tight_layout()
    plt.savefig(f"output/covariance_heatmap_time{target_hour}_2023.png")
    print(f"ヒートマップを 'covariance_heatmap_time{target_hour}_2023.png' として保存しました。")
    plt.show()

else:
    print(f"{target_hour}時の共分散行列を計算するための十分なデータがありません。")
