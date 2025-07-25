import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import (
    extract_consumer_name,
    load_and_clean_csv,
    filter_target_dates,
    make_pivot,
    is_complete_year_data
)

# 入力（ターミナルから時間帯を指定）
user_input = input("共分散行列を計算したい時間帯（0〜23）を入力してください（未入力なら12）: ")
target_hour = user_input if user_input else "12"
target_hour = target_hour.zfill(2)

# データ読み込み設定
# 設定
list_path = 'OPEN_DATA_60/list_60.csv'
data_dir = 'OPEN_DATA_60/raw'
target_start_md = '04-01'
target_end_md = '05-31'
target_days = 61
hours_per_day = 24
expected_rows = target_days * hours_per_day

# データ格納
data_matrix = []
consumer_names = []
excluded_files = []

# ファイルリスト読み込み
df_list = pd.read_csv(list_path, encoding='cp932')
df_list.columns = df_list.columns.str.strip()

# 各ファイル（各消費者）ごとに処理
for _, row in df_list.iterrows():
    file_name = row['ファイル名']
    consumer_name = extract_consumer_name(file_name)
    path = os.path.join(data_dir, file_name)

    df_raw = load_and_clean_csv(path)
    if df_raw is None:
        continue

    df_raw["年"] = df_raw["計測日"].dt.year
    file_has_valid_data = False

    for year in sorted(df_raw["年"].unique()):
        # その年の4月1日〜5月31日の期間を取得
        target_start = pd.to_datetime(f"{year}-{target_start_md}")
        target_end = pd.to_datetime(f"{year}-{target_end_md}")
        target_dates = pd.date_range(start=target_start, end=target_end)

        df_period = is_complete_year_data(df_raw, target_dates, expected_rows)
        if df_period is None:
            continue  # データ不完全な年はスキップ

        pivot = make_pivot(df_period)

        # 時間表記を2桁に揃える（例：'1' → '01'）
        pivot.index = pivot.index.astype(str).str.extract(r'(\d{1,2})')[0].astype(int).astype(str).str.zfill(2)

        if target_hour not in pivot.index or pivot.shape[1] != 61:
            continue  # データ不足

        data_matrix.append(pivot.loc[target_hour].values)
        consumer_names.append(f"{consumer_name} ({year})")
        file_has_valid_data = True

    if not file_has_valid_data:
        excluded_files.append(f"{file_name}（期間不一致またはデータ不足）")

# 結果出力
print(f"有効な消費者数（{target_hour}時）: {len(data_matrix)}")
print(f'\n除外されたファイルの数:', len(excluded_files))

# 共分散行列の計算と表示
if len(data_matrix) > 1:
    cov_matrix = np.cov(np.array(data_matrix), bias=True) # divided by n
    cov_df = pd.DataFrame(cov_matrix, index=consumer_names, columns=consumer_names)
    
    # CSVとして保存
    cov_df.to_csv(f"output/covariance_matrix_{target_hour}_all.csv", encoding='utf-8-sig')
    print(f"共分散行列を 'covariance_matrix_{target_hour}_all.csv' に保存しました。e")

    # ヒートマップとして保存
    plt.figure()
    sns.heatmap(cov_df, annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False, vmin=-25, vmax=25,
                cbar_kws={'label': 'Covariance'})
    plt.xlabel("Consumer")
    plt.ylabel("Consumer")
    plt.title(f"Covariance in the {target_hour} time slot")
    plt.tight_layout()
    plt.savefig(f"output/covariance_heatmap_all_{target_hour}_all.png")
    print(f"ヒートマップを 'covariance_heatmap_all_{target_hour}_all.png' として保存しました。")
    plt.show()

else:
    print(f"{target_hour}時の共分散行列を計算するための十分なデータがありません。")
