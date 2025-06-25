import pandas as pd
import os
import matplotlib.pyplot as plt

# list_60.csv を読み込み
df_list = pd.read_csv('OPEN_DATA/list_60.csv', encoding='cp932')
df_list.columns = df_list.columns.str.strip()

# 関東のデータだけ抽出
df_kanto = df_list[df_list['所在地'] == '関東']

# 探索ディレクトリ
data_dir = 'OPEN_DATA/raw'

# 結果を格納
all_data = []
target_date = '2013/06/01'  # ← 形式注意！（ゼロ埋めあり）

for _, row in df_kanto.iterrows():
    file_name = row['ファイル名']
    file_path = None

    # ファイルを raw / raw2 の中から探す
    path = os.path.join(data_dir, file_name)
    if os.path.isfile(path):
        file_path = path
        break

    if file_path is None:
        continue

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"読み込み失敗: {file_path}, 理由: {e}")
        continue

    # 必要な列が揃っているか確認
    if not {'計測日', '計測時間', '全体'}.issubset(df.columns):
        print(f"列が足りない: {file_path}")
        continue

    # 指定日のデータだけ抽出
    df_day = df[df['計測日'] == target_date]

    if df_day.empty:
        continue

    # 時間を index に、全体の電力使用量を value に
    hourly = df_day.set_index('計測時間')['全体']
    taall_data.append(hourly)

# 平均プロット
if not all_data:
    print("対象データが見つかりませんでした。")
else:
    df_all = pd.concat(all_data, axis=1)
    df_mean = df_all.mean(axis=1).sort_index()

    # プロット
    plt.figure(figsize=(10, 6))
    df_mean.plot(marker='o')
    plt.title('関東地域の平均電力使用量（全体） - 2013年6月1日')
    plt.xlabel('時間（1〜24）')
    plt.ylabel('平均使用量 [kWh]')
    plt.xticks(range(1, 25))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
