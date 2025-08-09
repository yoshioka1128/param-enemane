import pandas as pd
import matplotlib.pyplot as plt
import utils

# 入力CSVファイル名
input_file = "output/power_consumption_hourly_mixup_restricted.csv"

# CSV読み込み
df = pd.read_csv(input_file)

# "Original"で始まるConsumerだけ抽出
df_original = df[df["Consumer"].str.startswith("Original")]

# HourごとにMeanの平均とデータ数を計算
hourly_stats = df_original.groupby("Hour")["Mean"].agg(['mean', 'count']).reset_index()

# 列名をわかりやすく変更
hourly_stats.columns = ["Hour", "AvgMean", "Count"]

# データ数（1時間あたり）
count_per_hour = hourly_stats["Count"].iloc[0]  # すべてのHourで同じはず

# 出力ファイル名に件数を反映
output_file = f"output/hourly_consumer_mean_count{count_per_hour}.csv"

# CSVに書き出し
hourly_stats.to_csv(output_file, index=False)

# 図示
plt.figure()
plt.plot(hourly_stats["Hour"].to_numpy(), hourly_stats["AvgMean"].to_numpy(), marker='o')
plt.xlabel("Hour")
plt.ylabel("Average Consumption (kW)")
plt.title(f"Average Mean Consumption per Hour\n(Consumers starting with 'Original', {count_per_hour} per hour)")
plt.grid(True)
plt.xticks(range(1, 25))
plt.tight_layout()
plt.savefig(f"hourly_consumer_mean_count{count_per_hour}.png", dpi=300)
plt.show()

