import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import utils

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

# ファイルリスト読み込み
df_list = pd.read_csv(list_path, encoding='cp932')
df_list.columns = df_list.columns.str.strip()

for _, row in df_list.iterrows():
    file_name = row['ファイル名']
    consumer_name = file_name.replace('.csv', '')
    file_path = os.path.join(data_dir, file_name)

    if not os.path.isfile(file_path):
        continue

    try:
        df_raw = pd.read_csv(
            file_path,
            encoding='utf-8-sig',
            usecols=[0, 1, 2],
            header=0,
            names=["計測日", "計測時間", "全体"]
        )
        df_raw["計測日"] = pd.to_datetime(df_raw["計測日"], errors='coerce', format='%Y/%m/%d')
        df_raw = df_raw.dropna(subset=["計測日"])

        # 対象期間のデータに絞る
        df = df_raw[df_raw["計測日"].isin(target_dates)]
        if len(df) != 61 * 24:
            continue

        pivot = df.pivot(index='計測時間', columns='計測日', values='全体')
        
        # 時間表記を統一（例：'12:00' → '12'）
        pivot.index = pivot.index.astype(str).str.extract(r'(\d{1,2})')[0].astype(int).astype(str).str.zfill(2)

        if target_hour not in pivot.index or pivot.shape[1] != 61:
            continue

        data_matrix.append(pivot.loc[target_hour].values)
        consumer_names.append(consumer_name)

    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        continue

print(f"有効な消費者数（{target_hour}時）: {len(data_matrix)}")
#print("有効な消費者名:", len(consumer_names))

# 共分散行列の計算と表示
if len(data_matrix) > 1:
    cov_matrix = np.cov(np.array(data_matrix))
#    cov_matrix = np.cov(np.array(data_matrix), ddof=0)
    cov_df = pd.DataFrame(cov_matrix, index=consumer_names, columns=consumer_names)
    
    # CSVとして保存
    cov_df.to_csv(f"output/covariance_matrix_{target_hour}.csv", encoding='utf-8-sig')
    print(f"共分散行列を 'covariance_matrix_{target_hour}.csv' に保存しました。e")

    # ヒートマップとして保存
    plt.figure()
    sns.heatmap(cov_df, annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False, vmin=-25, vmax=25,
                cbar_kws={'label': 'Covariance'})
    plt.xlabel("Consumer")
    plt.ylabel("Consumer")
    plt.title(f"Covariance in the {target_hour} time slot")
    plt.tight_layout()
    plt.savefig(f"output/covariance_heatmap_{target_hour}.png")
    print(f"ヒートマップを 'covariance_heatmap_{target_hour}.png' として保存しました。")
    plt.show()

else:
    print(f"{target_hour}時の共分散行列を計算するための十分なデータがありません。")
