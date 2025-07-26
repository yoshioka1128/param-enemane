import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import seaborn as sns
from utils import (
    extract_consumer_name,
    load_and_clean_csv,
    filter_target_dates,
    make_pivot,
    is_complete_year_data,
    calc_hourly_stats
)

# --- 入力（ターミナルから時間帯を指定） ---
user_input = input("共分散行列を計算したい時間帯（0〜23）を入力してください（未入力なら12）: ")
target_hour = user_input if user_input else "12"
target_hour = target_hour.zfill(2)

# --- ファイル・パス・定数定義 ---
df_list = pd.read_csv('OPEN_DATA_60/list_60.csv', encoding='cp932')
df_list.columns = df_list.columns.str.strip()
data_dir = 'OPEN_DATA_60/raw'

target_start_md = '04-01'
target_end_md = '05-31'
target_days = 61
hours_per_day = 24
expected_rows = target_days * hours_per_day

# 契約電力区分ごとのプロファイル格納用辞書
consumer_profiles_by_contract = {'低圧': [], '高圧': [], '高圧小口': []}
excluded_files = []

# --- データ読み込み ---
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
    df_raw["計測時間"] = df_raw["計測時間"].astype(str).str.zfill(2)

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
        if target_hour not in pivot.index or pivot.shape[1] != target_days:
            continue  # データ不足

        if target_hour in pivot.index and pivot.shape[1] == target_days:
            y = pivot.loc[target_hour].values  # ← ここが変更の主眼、shape=(61,)
            consumer_profiles_by_contract[contract_type].append((None, y, None, consumer_name, year))
            file_valid = True
    
    if not file_valid:
        excluded_files.append(f"{file_name}（データ不足または対象期間なし）")

# --- Mixup 拡張 + 共分散計算 --------------------------------------------------
mixup_index = 1
data_matrix = [] # 全需要家（元＋合成）データ格納
all_names = []

random.seed(42)  # 再現性のため固定

for contract_type, profiles in consumer_profiles_by_contract.items():
    num_original = len(profiles)
    num_synthetic = int(num_original * 2.4)
    print(f"[{contract_type}] original: {num_original}, mixup: {num_synthetic}")

    # 元データをまず追加
    for p in profiles:
        data_matrix.append(p[1])
        all_names.append(f"{p[3]}_{p[4]}")  # 例: ConsumerA_2022

    # Mixup 合成
    for i in range(num_synthetic):
        if len(profiles) < 2:
            continue
        a, b = random.sample(profiles, 2)
        lam = random.uniform(0.3, 0.7)
        y_mix = lam * a[1] + (1 - lam) * b[1]
#        print(y_mix)
#        exit()

        a_name = f"{a[3]}_{a[4]}"
        b_name = f"{b[3]}_{b[4]}"
        label = f"Mixup_{mixup_index} ({contract_type}): {a_name} + {b_name}"
#        data_matrix.append(y_mix)
#        all_names.append(label)
        mixup_index += 1

# --- 共分散行列計算 ---
#profile_array =[]
#profile_array.append(all_profiles)  # shape: (num_profiles, 1)
cov_matrix = np.cov(np.array(data_matrix), ddof=0)
print("data_matrix.shape =", np.array(data_matrix).shape)
#print("example profile shape:", np.array(all_profiles[0]).shape)

# --- 保存と可視化 ---
os.makedirs("output", exist_ok=True)
cov_df = pd.DataFrame(cov_matrix, index=all_names, columns=all_names)
cov_df.to_csv(f"output/covariance_matrix_time{target_hour}_mixup_restricted.csv", encoding='utf-8-sig')
print(f"共分散行列を 'output/covariance_matrix_time{target_hour}_mixup_restricted.csv' に保存しました。")

plt.figure(figsize=(12, 10))
sns.heatmap(cov_df, cmap="coolwarm", xticklabels=False, yticklabels=False, vmin=-25, vmax=25, cbar_kws={"label": "Covariance"})
plt.title(f"Covariance Matrix with Mixup - Time {target_hour}")
plt.tight_layout()
plt.savefig(f"output/covariance_heatmap_time{target_hour}_mixup_restricted.png")
print(f"ヒートマップを 'output/covariance_heatmap_time{target_hour}_mixup_restricted.png' に保存しました。")
plt.show()
