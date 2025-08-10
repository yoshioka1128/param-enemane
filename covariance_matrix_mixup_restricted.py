import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import seaborn as sns
from utils import process_files

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
consumer_profiles_by_contract = {'低圧': [], '高圧小口': [], '高圧': []}
excluded_files = []

# --- データ読み込み ---
process_files(
    df_list, data_dir, target_start_md, target_end_md,
    expected_rows, target_days,
    consumer_profiles_by_contract,
    excluded_files,
    target_hour=target_hour
)

# --- Mixup 拡張 + 共分散計算 --------------------------------------------------
mixup_index = 1
original_index = 1
data_matrix = [] # 全需要家（元＋合成）データ格納
all_names = []

random.seed(42)  # 再現性のため固定

for contract_type, profiles in consumer_profiles_by_contract.items():
    num_original = len(profiles)
    num_synthetic = int(num_original * 2.4)
    print(f"[{contract_type}] original: {num_original}, mixup: {num_synthetic}")

    # 元データをまず追加
    for p in profiles:
        consumer_name = f"Original{original_index}_{p[1]}"
        original_index += 1
        data_matrix.append(p[0])
        all_names.append(consumer_name)  # 例: ConsumerA_2022

    # Mixup 合成
    for i in range(num_synthetic):
        if len(profiles) < 2:
            continue
        a, b = random.sample(profiles, 2)
        lam = random.uniform(0.3, 0.7)
        y_mix = lam * a[0] + (1 - lam) * b[0]
        label = f"Mixup{mixup_index}_{a[1]}_{b[1]}_lam={lam:.2f}"
        data_matrix.append(y_mix)
        all_names.append(label)
        mixup_index += 1

# --- 共分散行列計算 ---
cov_matrix = np.cov(np.array(data_matrix), ddof=0)
print("data_matrix.shape =", np.array(data_matrix).shape)

# --- 保存と可視化 ---
os.makedirs("output", exist_ok=True)
cov_df = pd.DataFrame(cov_matrix, index=all_names, columns=all_names)
cov_df.to_csv(f"output/covariance_matrix_time{target_hour}_mixup_restricted.csv", encoding='utf-8-sig')
print(f"共分散行列を 'output/covariance_matrix_time{target_hour}_mixup_restricted.csv' に保存しました。")

plt.figure()
sns.heatmap(cov_df, annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False, vmin=-5, vmax=5,
            cbar_kws={'label': 'Covariance'})

plt.title(f"Covariance Matrix with Mixup - Time {target_hour}")
plt.xlabel("Consumer")
plt.ylabel("Consumer")
plt.tight_layout()
plt.savefig(f"output/covariance_heatmap_time{target_hour}_mixup_restricted.png")
print(f"ヒートマップを 'output/covariance_heatmap_time{target_hour}_mixup_restricted.png' に保存しました。")
#plt.show()
