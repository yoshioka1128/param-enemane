import pandas as pd
import matplotlib.pyplot as plt
import os

# データリスト読み込み
df_list = pd.read_csv('OPEN_DATA/list_60.csv', encoding='cp932')
df_list.columns = df_list.columns.str.strip()
df_kanto = df_list[df_list['所在地'] == '関東']

data_dir = 'OPEN_DATA/raw'
target_dates = pd.date_range(start='2013-04-01', end='2013-05-31')
target_date_strs = target_dates.strftime('%Y/%m/%d')

plt.figure()
output_rows = []

for idx, row in df_kanto.iterrows():
    file_name = row['ファイル名']
    consumer_name = file_name.replace('.csv', '')

    path = os.path.join(data_dir, file_name)
    if not os.path.isfile(path):
        continue

    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
        df = df[df['計測日'].isin(target_date_strs)]
#        if len(df) == 61 * 24:
#            print(f"{consumer_name}: matched rows = {len(df)}")
        if len(df) != 61 * 24:
            continue
        
        pivot = df.pivot(index='計測時間', columns='計測日', values='全体')
        hourly_mean = pivot.mean(axis=1)
        hourly_std = pivot.std(axis=1)

        x = hourly_mean.index.astype(int).values
        y = hourly_mean.values
        yerr = hourly_std.values

        # グラフ描画
        plt.plot(x, y, label=consumer_name)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.1)

        # 出力用データ蓄積
        for h, m, s in zip(x, y, yerr):
            output_rows.append({'Consumer': consumer_name, 'Hour': int(h), 'Mean': m, 'Std': s})

    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        continue

# グラフ保存
plt.xlabel('Hour of Day')
plt.ylabel('Predicted Negawatt [kWh]')
plt.title('Predicted Negawatt on June 1')
plt.grid(True)
plt.savefig("output/predicted_june1_kanto_all.png")
plt.show()
plt.close()

# CSV出力
output_df = pd.DataFrame(output_rows)
os.makedirs('output', exist_ok=True)
output_df.to_csv('output/predicted_negawatt_hourly_stats.csv', index=False)

