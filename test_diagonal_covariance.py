import pandas as pd
import numpy as np

# 読み込むファイル名（必要に応じて変更してください）
user_input = input("共分散行列を計算したい時間帯（0〜23）を入力してください（未入力なら12）: ")
target_hour = user_input if user_input else "12"
target_hour = target_hour.zfill(2)

input_filename = f"output/covariance_matrix_time{target_hour}_mixup_restricted.csv"
#output_filename = f"output/diagonal_covariance_time{target_hour}.csv"

# CSVファイルをDataFrameとして読み込む（インデックス列を指定）
cov_df = pd.read_csv(input_filename, index_col=0)

# 対角成分（自己共分散）を抽出
diagonal_series = cov_df.loc[cov_df.index, cov_df.columns].apply(lambda row: row[row.name], axis=1)
print('diagonal_series')
print(diagonal_series)

# --- 予測データ（平均と標準偏差）の読み取り ---
stats_path = 'output/predicted_negawatt_hourly_stats_mixup_restricted.csv'
stats_df = pd.read_csv(stats_path)
# 対象時間のみにフィルター
hour_df = stats_df[stats_df['Hour'] == int(target_hour)][['Consumer', 'Std']].copy()

# 標準偏差から分散を計算: (Std)^2
hour_df['Variance_Std'] = (hour_df['Std'] ** 2)

print('hour_df')
print(hour_df)

# リスト化（順番あり）
cov_list = list(diagonal_series.index)
stats_list = list(hour_df['Consumer'])

# 長さを比較
print(f"共分散データ: {len(cov_list)} 件")
print(f"統計データ: {len(stats_list)} 件")

# 順番が完全に一致するか？
if cov_list == stats_list:
    print("Consumer の順番も完全に一致しています。")
else:
    print("Consumer の順番に違いがあります。")

    # 一致しない場所の例を表示（最初の10件だけ）
    print("\n≠ 順番の不一致（最初の10件）:")
    for i, (a, b) in enumerate(zip(cov_list, stats_list)):
        if a != b:
            print(f"{i}: cov='{a}', stats='{b}'")
        if i >= 9:
            break

# diagonal_series をリスト化
cov_values = diagonal_series.tolist()
# hour_df の分散値をリスト化
stats_values = hour_df['Variance_Std'].tolist()
# Consumer名も取得
consumer_list = diagonal_series.index.tolist()

# 比較結果を表示（最初の10件）
print("=== 順番付きで値の比較（最初の10件）===")
for i, (name, cov, var) in enumerate(zip(consumer_list, cov_values, stats_values)):
    diff = cov - var
    print(f"{i}: {name} | Cov: {cov:.6f} | Var: {var:.6f} | Diff: {diff:.6f}")
    if i >= 9:
        break

print('cov_values')
print(np.array(cov_values))
print(np.isnan(np.array(cov_values)).sum())
print('stats_vaues')
print(np.array(stats_values))
print(np.isnan(np.array(stats_values)).sum())

# NaN を含む consumer を抽出
nan_consumers = []
for name, var in zip(consumer_list, stats_values):
    if pd.isna(var):
        nan_consumers.append(name)

# 結果を出力
print("\n=== NaN を含む Consumer 一覧 in negawatt ===")
if nan_consumers:
    for c in nan_consumers:
        print(c)
    print(f"\n合計: {len(nan_consumers)} 件の Consumer に NaN が含まれています。")
else:
    print("NaN を含む Consumer は存在しません。")
exit()

# 一致している割合（ある程度許容誤差あり）
diffs = np.abs(np.array(cov_values) - np.array(stats_values))
tolerance = 1e-6
num_close = np.sum(diffs < tolerance)
print(f"値が {tolerance} 以下の差で一致している数: {num_close}/{len(cov_values)}")
print(f"NaN in diffs: {np.isnan(diffs).sum()}")  # NaNが何個あるか

# 一致していないインデックスを抽出
not_matching_indices = np.where(diffs >= tolerance)[0]

# 対応するConsumer名を抽出
not_matching_consumers = [consumer_list[i] for i in not_matching_indices]

# 結果出力
print(f"一致していない Consumer 数: {len(not_matching_consumers)} / {len(consumer_list)}")
print("=== 一致していない Consumer の例（最大10件）===")
for name in not_matching_consumers[:10]:
    print(name)
