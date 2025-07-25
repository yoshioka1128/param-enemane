import pandas as pd
import os
import matplotlib.pyplot as plt

df_list = pd.read_csv('OPEN_DATA_60/list_60.csv', encoding='cp932')
df_list.columns = df_list.columns.str.strip()
df_kanto = df_list[df_list['所在地'] == '関東']

dirs = ['OPEN_DATA_60/raw']
all_data = []
target_dates = pd.date_range('2013-04-01', '2013-05-31').strftime('%Y/%m/%d')
predict_date = '2013/06/01'

for _, row in df_kanto.iterrows():
    file_name = row['ファイル名']
    file_path = None
    for d in dirs:
        path = os.path.join(d, file_name)
        if os.path.isfile(path):
            file_path = path
            break

    if file_path is None:
        continue

    try:
        df = pd.read_csv(
            file_path,
            encoding='utf-8-sig',
            usecols=[0, 1, 2],
            header=0,
            names=["計測日", "計測時間", "全体"]
        )
    except:
        continue

    if not {'計測日', '計測時間', '全体'}.issubset(df.columns):
        continue

    # フィルタ対象期間のデータ取得
    df_range = df[df['計測日'].isin(target_dates)]
    if df_range.empty:
        continue

    pivot = df_range.pivot(index='計測日', columns='計測時間', values='全体')
    all_data.append(pivot)

print('ファイル数', len(all_data))
    
# 全体データ結合（日付 x 時間）
if not all_data:
    print("予測用データが見つかりませんでした。")
else:
    df_all = pd.concat(all_data)
    df_mean = df_all.groupby(df_all.index).mean()  # 同日が複数あれば平均
    hourly_mean = df_mean.mean(axis=0)
    hourly_std = df_mean.std(axis=0, ddof=0)

    # 予測結果プロット
    plt.figure(figsize=(10, 6))
    x = hourly_mean.index.astype(int).values  # 1〜24

    plt.plot(x, hourly_mean.values, label='Predicted Average (4/1-5/31)', marker='o')
    plt.fill_between(x,
                     (hourly_mean - hourly_std).values,
                     (hourly_mean + hourly_std).values,
                     alpha=0.3, label=r'$\pm$ $\sigma$')
    
    # 実測（6月1日）があれば追加プロット
    actual_all = []
    for _, row in df_kanto.iterrows():
        file_name = row['ファイル名']
        for d in dirs:
            path = os.path.join(d, file_name)
            if os.path.isfile(path):
                try:
                    df = pd.read_csv(path, encoding='utf-8-sig')
                    df_day = df[df['計測日'] == predict_date]
                    if not df_day.empty:
                        df_day = df_day.set_index('計測時間')['全体']
                        actual_all.append(df_day)
                except:
                    continue
                break

    if actual_all:
        actual_df = pd.concat(actual_all, axis=1)
        actual_mean = actual_df.mean(axis=1).sort_index()
        plt.plot(actual_mean.index.values, actual_mean.values, label='Measured (6/1)',
                 linestyle='--', marker='x', color='red')

    plt.title('Electricity demand forecast for June 1 (Kanto): moving average'+r'$\pm$ $\sigma$')
    plt.xlabel('Time')
    plt.ylabel('Average usage [kWh]')

    plt.xticks(x)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
