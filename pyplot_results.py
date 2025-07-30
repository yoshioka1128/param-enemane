import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# データ読み込み
df_result = pd.read_csv('output/optimal_consumer_combination.csv')
df_pred = pd.read_csv('output/predicted_negawatt_hourly_stats.csv')

# 結果格納用リスト
hours = []
procured_values = []
total_means = []
total_stds = []

# 定数：標本数（61日）
n = 61

# 各時間帯ごとに集計
for _, row in df_result.iterrows():
    hour = int(row['Hour'])
    procured = row['Procured']

    # NaN の場合はスキップ
    if pd.isna(row['Selected_Consumers']):
        print(f"Hour {hour} は選択された需要家が存在しません。スキップします。")
        continue

    selected_consumers = row['Selected_Consumers'].split(';')

    # 対象時間帯のデータから選択された需要家のみ抽出
    df_hour = df_pred[df_pred['Hour'] == hour]
    df_selected = df_hour[df_hour['Consumer'].isin(selected_consumers)]

    # 合計平均
    total_mean = df_selected['Mean'].sum()

    # 分散を加算（標準偏差が母標準偏差なら、単純に2乗でOK）
    variances = df_selected['Std'] ** 2  # 各需要家の分散（母分散とみなす）
    total_variance = variances.sum()     # 独立変数の分散の和
    total_std = np.sqrt(total_variance)  # 合成標準偏差
    
    # リストに追加
    hours.append(hour)
    procured_values.append(procured)
    total_means.append(total_mean)
    total_stds.append(total_std)

# プロット
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# カラーマップから色を適当に割り当て（例: tab10を繰り返す）
#cmap = cm.get_cmap('tab10')
cmap = plt.colormaps.get_cmap('tab10')
color_cycle = [cmap(i % 10) for i in range(100)]  # 最大100色（繰り返し）
color_idx = 0  # 色のインデックス管理用

full_hours = list(range(1, 25))
procured_dict = dict(zip(hours, procured_values))
procured_filled = [procured_dict.get(h, 0) for h in full_hours]
#plt.figure(figsize=(10, 6))
plt.plot(full_hours, procured_filled, color='tab:blue', linestyle='-', label='Target Procured')
plt.errorbar(hours, total_means, yerr=total_stds, fmt='o', linestyle='--', color='red', label=r'Negawatt $\pm$ $\sigma$', capsize=5)

# 各需要家の平均値ラインを追加
for _, row in df_result.iterrows():
    hour = int(row['Hour'])
    if pd.isna(row['Selected_Consumers']):
        continue
    selected_consumers = row['Selected_Consumers'].split(';')
    for consumer in selected_consumers:
        df_indiv = df_pred[(df_pred['Hour'] == hour) & (df_pred['Consumer'] == consumer)]
        if not df_indiv.empty:
            mean_val = df_indiv['Mean'].values[0]
#            plt.plot(hour, mean_val, marker='x', color=color_cycle[color_idx % len(color_cycle)], alpha=0.6)
            color_idx += 1

# 軸とラベル
plt.xlabel('Time')
plt.ylabel('Negawatt [kW]')
plt.title('Negawatt vs Target')
plt.xticks(full_hours)
plt.grid(True)
plt.legend()
#plt.ylim(0, 1000)
plt.xlim(0, 24)

# 保存
os.makedirs('output', exist_ok=True)
plt.savefig('output/consumer_selection_plot.png', dpi=300)
plt.show()
plt.close()

print("output/consumer_selection_plot.png を出力しました。")





print("Hour\tSelected_Count\tPer_Consumer[kW]\tProcured[kW]")

for hour in range(1, 25):
    row = df_result[df_result['Hour'] == hour]
    if row.empty or pd.isna(row.iloc[0]['Selected_Consumers']):
        count = 0
        procured = 0
        per_consumer = 0
    else:
        selected = row.iloc[0]['Selected_Consumers'].split(';')
        count = len(selected)
        procured = row.iloc[0]['Procured']
        per_consumer = procured / count if count > 0 else 0

    print(f"{hour:>2}\t{count:>14}\t{per_consumer:>15.2f}\t{procured:>12.2f}")

    


# 時間ごとに選ばれた需要家の Mean を積み上げ棒グラフにする
fig, ax = plt.subplots()

consumer_bar_data = {}  # consumer別の時間ごとのデータ
unique_consumers = set()

for _, row in df_result.iterrows():
    hour = int(row['Hour'])
    if pd.isna(row['Selected_Consumers']):
        continue

    selected_consumers = row['Selected_Consumers'].split(';')
    df_hour = df_pred[df_pred['Hour'] == hour]
    for consumer in selected_consumers:
        df_c = df_hour[df_hour['Consumer'] == consumer]
        if not df_c.empty:
            mean_val = df_c['Mean'].values[0]
            key = consumer
            unique_consumers.add(consumer)
            if key not in consumer_bar_data:
                consumer_bar_data[key] = [0] * 24  # index: 0〜23 → hour 1〜24
            consumer_bar_data[key][hour - 1] = mean_val  # hour は1始まり

# スタック棒グラフの作成
bottoms = np.zeros(24)
for idx, consumer in enumerate(sorted(consumer_bar_data.keys())):
    values = consumer_bar_data[consumer]
    ax.bar(range(1, 25), values, bottom=bottoms, color=color_cycle[idx % len(color_cycle)],
#           edgecolor='white', linewidth=0.3, label=consumer if idx < 10 else None)  # 凡例は10個まで
           edgecolor='white', linewidth=0.3)  # 凡例はなし    
    bottoms += np.array(values)

# total_mean を重ねて表示（確認用）
#ax.plot(hours, total_means, 'o--', color='black', label='Total Mean')
ax.plot(full_hours, procured_filled, color='tab:blue', linestyle='-', label='Target Procured')

ax.set_xlabel('Time')
ax.set_ylabel('Negawatt [kW]')
ax.set_title('Stacked Bar of Selected Consumers per Hour')
ax.set_xticks(range(1, 25))
ax.grid(True)
ax.set_xlim(0.5, 24.5)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('output/stacked_bar_selected_consumers.png', dpi=300)
plt.show()
plt.close()

print("output/stacked_bar_selected_consumers.png を出力しました。")
