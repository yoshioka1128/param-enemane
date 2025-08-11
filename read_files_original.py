import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import utils

# ---------- 平均値の計算 ----------
input_file = "param/power_consumption_hourly_mixup_restricted.csv"
df = pd.read_csv(input_file)

files = sorted(glob.glob("output/covariance_matrix_time*_mixup_restricted.csv"))
stats_list = []
count_per_hour = None

for file in files:
    match = re.search(r"time(\d+)_", file)
    if not match:
        continue
    hour = int(match.group(1))

    # 共分散行列読み込みとOriginal抽出
    cov_df = pd.read_csv(file, index_col=0)
    original_cols = [c for c in cov_df.columns if c.startswith("Original")]
    cov_df_original = cov_df.loc[original_cols, original_cols]

    # dfから同じHourのOriginalだけ抽出して平均値
    df_hour_original = df[(df["Hour"] == hour) & (df["Consumer"].str.startswith("Original"))]

    # ここでチェック関数を呼ぶ
    utils.check_variance_match_single(hour, cov_df_original, df_hour_original, original_cols)

    # 平均値&データ数
    avg_mean = df_hour_original["Mean"].mean()
    count_this_hour = len(df_hour_original)
    if count_per_hour is None:
        count_per_hour = count_this_hour  # 最初の時間の件数を記録

    # 標準偏差計算
    cov_matrix = cov_df_original.to_numpy()
    N = cov_matrix.shape[0]
    ones = np.ones((N, 1))
    var_mean = (ones.T @ cov_matrix @ ones)[0, 0] / (N**2)
    std_mean = np.sqrt(var_mean)

    stats_list.append({"Hour": hour, "AvgMean": avg_mean, "StdMean": std_mean})

# CSV出力
stats_df = pd.DataFrame(stats_list).sort_values("Hour")
output_file = f"output/hourly_consumer_mean_std_count{count_per_hour}.csv"
stats_df.to_csv(output_file, index=False)

# ===== プロット =====
hours = stats_df["Hour"].to_numpy()
avg_means = stats_df["AvgMean"].to_numpy()
std_means = stats_df["StdMean"].to_numpy()

plt.figure()
plt.subplots_adjust(left=0.1, right=0.98)
plt.plot(hours, avg_means, marker='o', color='blue', label="Average Value")
plt.fill_between(hours,
                 avg_means - std_means,
                 avg_means + std_means,
                 color='blue', alpha=0.2, label=r"$\pm\sigma_t$")
plt.xlabel("Time")
plt.xlim(1, 24)
plt.xticks(range(1, 25))
plt.ylim(0, 80)
plt.ylabel("Power Consumption [kWh]")
plt.title(f"{count_per_hour} consumers")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"output/hourly_consumer_mean_std_count{count_per_hour}.png")
plt.show()

