import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def parse_hour_range(input_str):
    """例: '12,13,14' → [12, 13, 14]"""
    try:
        hours = list(sorted(set(int(h.strip()) for h in input_str.split(",") if h.strip().isdigit())))
        return [h for h in hours if 0 <= h <= 23]
    except:
        print("時間帯の入力が無効です。カンマ区切りの整数を入力してください（例：12,13,14）")
        return []

def process_consumer_file(file_name, data_dir, start_date, end_date, target_hours):
    """
    個別の需要家CSVファイルを読み込み、対象期間・時間のデータ（183点）を返す。
    条件に満たない場合は None を返す。
    """
    path = os.path.join(data_dir, file_name)
    if not os.path.isfile(path):
        print(f"{file_name}: file not found")
        return None

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")

        # 必要な列の存在チェック
        required_columns = {"計測日", "計測時間", "全体"}
        if not required_columns.issubset(df.columns):
            print(f"{file_name}: required columns not found")
            return None

        df["計測日"] = pd.to_datetime(df["計測日"], errors="coerce")

        if df["計測日"].isna().all():
            print(f"{file_name}: all 計測日 values are NaT")
            return None

        df = df[df["計測日"].between(start_date, end_date)]
        if df.empty:
            print(f"{file_name}: no data in date range")
            return None

        df = df[df["計測時間"].isin(target_hours)]
        if df.empty:
            print(f"{file_name}: no data at 17–19 o'clock")
            return None

        df_sorted = df.sort_values(["計測日", "計測時間"])
        values = df_sorted["全体"].values

        if len(values) == 183:
            return values
        else:
            print(f"Skipping {file_name}: expected 183 values, got {len(values)}")
            return None

    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        return None

def detect_nan_pairs(cov_matrix):
    """共分散がNaNのペアを検出してリストで返す"""
    nan_pairs = []
    consumers = cov_matrix.columns
    for i in range(len(consumers)):
        for j in range(i + 1, len(consumers)):
            if pd.isna(cov_matrix.iloc[i, j]):
                nan_pairs.append((consumers[i], consumers[j]))
    return nan_pairs


def save_covariance_matrix(cov_matrix, hour_label, output_dir="output"):
    """共分散行列をPNGとCSVで保存"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if cov_matrix.isna().all().all():
        print("共分散行列が全て NaN のため、出力をスキップしました。")
        return

    png_file = os.path.join(output_dir, f"covariance_matrix_{hour_label}.png")
    csv_file = os.path.join(output_dir, f"covariance_matrix_{hour_label}.csv")

    # プロット
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cov_matrix,
        cmap="coolwarm",
        vmin=-50,
        vmax=50,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Covariance'}
    )
    plt.title(f"Covariance Matrix ({hour_label}, Apr–May 2013)", fontsize=14)
    plt.xlabel("Consumer")
    plt.ylabel("Consumer")
    plt.tight_layout()
    plt.savefig(png_file, dpi=300)
    plt.close()

    # CSV保存
    cov_matrix.to_csv(csv_file)

    print(f"共分散行列を保存しました: {png_file}, {csv_file}")

def main():
    # ユーザーから時間帯の入力を受け取る
    user_input = input("対象時間帯（カンマ区切り, 例: 12,13,14）を入力してください: ")
    target_hours = parse_hour_range(user_input)

    if not target_hours:
        print("対象時間帯が正しく指定されていません。終了します。")
        return

    start_date = pd.to_datetime("2013-04-01")
    end_date = pd.to_datetime("2013-05-31")

    df_list = pd.read_csv("OPEN_DATA/list_60.csv", encoding="cp932")
    df_list.columns = df_list.columns.str.strip()
    df_kanto = df_list[df_list["所在地"] == "関東"]
    file_names = df_kanto["ファイル名"].values
    data_dir = 'OPEN_DATA/raw'

    consumer_data = {}
    for file_name in file_names:
        values = process_consumer_file(file_name, data_dir, start_date, end_date, target_hours)
        if values is not None:
            consumer_data[file_name.replace(".csv", "")] = values

    df_matrix = pd.DataFrame(consumer_data)
    cov_matrix = df_matrix.cov()

    # NaNペアを検出して表示
    nan_pairs = detect_nan_pairs(cov_matrix)
    if nan_pairs:
        print("共分散が NaN となった需要家ペア:")
        for a, b in nan_pairs:
            print(f" - {a} × {b}")
    else:
        print("すべての需要家ペアで共分散が計算されました。")

    # 時間帯ラベル作成（例：hour12_14）
    if len(target_hours) > 1 and all(np.diff(target_hours) == 1):
        hour_label = f"hour{target_hours[0]}_{target_hours[-1]}"
    else:
        hour_label = "hour" + "_".join(str(h) for h in target_hours)

    save_covariance_matrix(cov_matrix, hour_label)

if __name__ == "__main__":
    main()
