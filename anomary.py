import pandas as pd

# 出力された統計CSVを読み込む
df = pd.read_csv('output/predicted_negawatt_hourly_stats_all_years.csv')

# 閾値を設定（例: 標準偏差が12を超えるもの）
threshold = 100

# 異常と判断される行を抽出
anomalies = df[df['Std'] > threshold]

# 異常が含まれるファイル名（Consumer, Yearの組）をリストアップ
abnormal_files = anomalies[['Consumer', 'Year', 'Std']].drop_duplicates()

# 表示
print("異常に標準偏差が大きいデータを含むファイル（Consumer, Year）:")
print(abnormal_files.to_string(index=False))
