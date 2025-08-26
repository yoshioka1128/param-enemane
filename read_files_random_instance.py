import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import re
import random
import utils

def get_random_consumers(L, rng):
    input_file = f"./param/power_consumption_hourly_mixup_restricted.csv"
    df = pd.read_csv(input_file)
    original_consumers = df[df['Consumer'].str.startswith('Original')]['Consumer'].unique().tolist()
    consumers_list = rng.choice(original_consumers, size=L, replace=False)
    return consumers_list

def plot_selected_consumers_time_series(df, selected_originals, output_path):
    """
    選択した需要家の Mean ± Std の時系列を1枚のグラフに描画して保存する

    Parameters
    ----------
    df : pandas.DataFrame
        データフレーム (Consumer, Hour, Mean, Std を含む)
    selected_originals : list
        対象となる需要家のリスト
    output_path : str
        保存先ファイルパス (例: "output/selected_consumers_time_series.png")
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplots_adjust(left=0.1, right=0.98)
    nodes = len(selected_originals)
    # カラーマップ（20色程度を繰り返し使用）
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(nodes)]

    for idx, consumer in enumerate(selected_originals):
        df_c = df[df["Consumer"] == consumer].sort_values("Hour")
        hours = df_c["Hour"].to_numpy()
        means = df_c["Mean"].to_numpy() / 10
        stds = df_c["Std"].to_numpy() / 10

        color = colors[idx % len(colors)]

        # 各消費者の平均値を線で
        plt.plot(hours, means, color=color)

        # ±Std を塗りつぶし
        plt.fill_between(hours, means - stds, means + stds, alpha=0.1, color=color)

    plt.xlabel("Time")
    plt.xlim(1, 25)
    plt.xticks(range(1, 25))
    plt.ylim(bottom=0)
    plt.ylabel("Negawatt per Consumer [kWh]")
    plt.title(f"{nodes} consumers")
    plt.grid(True)

    # 凡例は無し
    plt.tight_layout()

    plt.savefig(output_path)
    plt.show()
    plt.close()

# ---------- パラメータ ----------
L = int(input("input problem size: (756)") or 756)
# random number
iseed = int(input("seed for initial angles (42): ") or 42)
rng = np.random.default_rng(iseed) # 乱数の種固定

#seed = int(input("input random number seed: (42)") or 42)
#random.seed(seed)  # 再現性確保のための乱数シード

# ---------- 平均値の計算 ----------
input_file = "param/power_consumption_hourly_mixup_restricted.csv"
df = pd.read_csv(input_file)

files = sorted(glob.glob("param/covariance_matrix_time*_mixup_restricted.csv"))

# 共分散行列ファイルの先頭から需要家リストを取得
if not files:
    raise FileNotFoundError("共分散行列ファイルが見つかりません。")

cov_df_first = pd.read_csv(files[0], index_col=0)
all_original_cols = [c for c in cov_df_first.columns if c.startswith("Original")]
original_consumers = df[df['Consumer'].str.startswith('Original')]['Consumer'].unique().tolist()
utils.compare_lists(all_original_cols, original_consumers)
print('maximum problem size', len(all_original_cols))

if L > len(all_original_cols):
    raise ValueError(f"L={L} は需要家数 {len(all_original_cols)} を超えています。")

# ---- ループ前に一度だけ選択 ----
#selected_originals = random.sample(all_original_cols, L)
selected_originals = get_random_consumers(L, rng)
df_selected_originals = pd.DataFrame(selected_originals, columns=['Consumer'])
df_selected_originals.to_csv(f"output/selected_originals_L{L}_seed{iseed}.csv", index=False)
print(f"output/selected_originals_L{L}_seed{iseed}.csv に書き出しました。")

plot_selected_consumers_time_series(
    df, selected_originals,
    f"output/procure_hourly_{L}consumers.png"
)

stats_list = []
count_per_hour = None

for file in files:
    match = re.search(r"time(\d+)_", file)
    if not match:
        continue
    hour = int(match.group(1))

    # 共分散行列を選択需要家に絞る
    cov_df = pd.read_csv(file, index_col=0)
    cov_df_selected = cov_df.loc[selected_originals, selected_originals]

    # dfから同じHour & 選択需要家だけ抽出
    df_hour_selected = df[(df["Hour"] == hour) & (df["Consumer"].isin(selected_originals))]

    # チェック関数呼び出し
    utils.check_variance_match_single(hour, cov_df_selected, df_hour_selected, selected_originals)

    # 平均値 & データ数
    avg_mean = df_hour_selected["Mean"].mean()
    count_this_hour = len(df_hour_selected)
    if count_per_hour is None:
        count_per_hour = count_this_hour  # 最初の時間の件数を記録

    # 標準偏差計算
    cov_matrix = cov_df_selected.to_numpy()
    N = cov_matrix.shape[0]
    ones = np.ones((N, 1))
    var_mean = (ones.T @ cov_matrix @ ones)[0, 0] / (N**2)
    std_mean = np.sqrt(var_mean)

    stats_list.append({"Hour": hour, "AvgMean": avg_mean, "StdMean": std_mean})

# ---------- 結果をDataFrame化 ----------
stats_df = pd.DataFrame(stats_list).sort_values("Hour")
output_file = f"output/hourly_consumer_mean_std_L{L}_seed{iseed}.csv"
stats_df.to_csv(output_file, index=False)

# ---------- プロット ----------
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
plt.ylim(bottom=0, top=None)
plt.ylabel("Power Consumption [kWh]")
plt.title(rf"random seed: {iseed} for $L=${L}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"output/hourly_consumer_mean_std_L{L}_seed{iseed}.png")
#plt.show()

