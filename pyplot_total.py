import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

df_list = pd.read_csv('OPEN_DATA/list_60.csv', encoding='cp932')
df_list.columns = df_list.columns.str.strip()
df_kanto = df_list[df_list['所在地'] == '関東']

data_dir = 'OPEN_DATA/raw'
all_data = []
target_dates = pd.date_range('2013-04-01', '2013-05-31').strftime('%Y/%m/%d')
predict_date = '2013/06/01'

valid_file_count = 0
for _, row in df_kanto.iterrows():
    file_name = row['ファイル名']
    file_path = os.path.join(data_dir, file_name)

    if not os.path.isfile(file_path):
        continue

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        continue

    if not {'計測日', '計測時間', '全体'}.issubset(df.columns):
        continue

    df_range = df[df['計測日'].isin(target_dates)]

#    if df_range.empty:
#        continue
    if len(df_range) != 61 * 24:
        continue
    valid_file_count += 1

    pivot = df_range.pivot(index='計測日', columns='計測時間', values='全体')
    all_data.append(pivot)

print(f"計算対象となったファイル数: {valid_file_count}")
print('len(all_data)', len(all_data))



# 全体データ結合（日付 x 時間）
if not all_data:
    print("予測用データが見つかりませんでした。")
else:
    df_all = pd.concat(all_data)
    df_mean = df_all.groupby(df_all.index).mean()
    hourly_total = df_mean.sum(axis=0)
    hourly_std = df_mean.std(axis=0)
#    df_mean = df_all.groupby(df_all.index).mean()  # 同日が複数あれば平均
#    hourly_mean = df_mean.mean(axis=0)
#    hourly_std = df_mean.std(axis=0)

    # ユーザー指定：対象時間帯と削減割合
    default_hours = [12, 13, 14, 15, 16, 17]
    default_ratio = 0.8
    input_hours = input("削減対象の時間帯（カンマ区切り 例: 12,13,14）を入力してください: ")
    input_hours = input_hours or "12,13,14,15,16,17"
    input_ratio = input("削減割合（例: 0.8 = 80%に削減）を入力してください: ")
    input_ratio = float(input_ratio or "0.8")

    try:
        target_hours = sorted(set(int(h.strip()) for h in input_hours.split(",") if h.strip().isdigit()))
        ratio = float(input_ratio)
        if not 0 <= ratio <= 1:
            raise ValueError
    except:
        print("入力が不正です。対象時間帯は整数、割合は0〜1で指定してください。")
        target_hours = []
        ratio = 1.0  # 実質無効
        
    # 削減後の予測データ作成
    reduced_total = hourly_total.copy()
    reduced_diff = pd.Series(0.0, index=hourly_total.index)

    for h in target_hours:
        if h in reduced_total.index:
            original_value = reduced_total[h]
            reduced_value = original_value * ratio
            reduced_diff[h] = original_value - reduced_value
            reduced_total[h] = reduced_value

    # プロット
    x = hourly_total.index.astype(int).values
    plt.plot(x, hourly_total.values, label='Predicted Total', marker='o')
    plt.fill_between(x,
                     (hourly_total - hourly_std).values,
                     (hourly_total + hourly_std).values,
                     alpha=0.3, label=r'$\pm \sigma$')

    plt.plot(x, reduced_total.values, label=f'Reduced (ratio={ratio})', linestyle='--', marker='s', color='green')
    plt.plot(x, reduced_diff.values, label='Reduction Amount', linestyle=':', marker='x', color='orange')

    # 実測（6月1日）があれば追加プロット
    actual_all = []
    for _, row in df_kanto.iterrows():
        file_name = row['ファイル名']
        path = os.path.join(data_dir, file_name)
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path, encoding='utf-8-sig')
                df_day = df[df['計測日'] == predict_date]
                if not df_day.empty:
                    df_day = df_day.set_index('計測時間')['全体']
                    actual_all.append(df_day)
            except:
                continue

#    if actual_all:
#        actual_df = pd.concat(actual_all, axis=1)
#        actual_mean = actual_df.mean(axis=1).sort_index()
#        plt.plot(actual_mean.index.values, actual_mean.values, label='Measured (6/1)',
#                 linestyle='-', marker='D', color='red')

    plt.title('Electricity Demand Forecast for June 1')
    plt.xlabel('Time')
    plt.ylabel('Average Usage [kWh]')
    plt.xticks(x)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 削減情報をファイル名に反映して保存
    # 時間帯ラベル作成（例：hour12_14）
    if len(target_hours) > 1 and all(np.diff(target_hours) == 1):
        hour_label = f"hour{target_hours[0]}_{target_hours[-1]}"
    else:
        hour_label = "hour" + "_".join(str(h) for h in target_hours)
    ratio_str = f"{ratio:.2f}".replace('.', '_')
    filename = f"output/reduced_{hour_label}_ratio{ratio_str}.png"
    plt.savefig(filename, dpi=300)
    print(f"{filename} に保存しました。")

    # 削減電力量をCSV出力
    output_df = pd.DataFrame({
        'Hour': reduced_diff.index.astype(int),
        'Procured': reduced_diff.values
    })
    csv_filename = f"output/procured_negawatt_{hour_label}_ratio{ratio_str}.csv"
    output_df.to_csv(csv_filename, index=False)
    print(f"{csv_filename} に保存しました。")
    
    # 表示（必要に応じて）
    plt.show()
