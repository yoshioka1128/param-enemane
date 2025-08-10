import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{mathptmx}",  # Times Roman を使う
    "font.size": 16,             # 全体フォントサイズ
    "axes.labelsize": 16,        # x/yラベル
    "xtick.labelsize": 14,       # x軸目盛
    "ytick.labelsize": 14,       # y軸目盛
    "legend.fontsize": 14,       # 凡例
    "axes.titlesize": 18,        # タイトル
})

import csv

def get_unique_consumers(file_path):
    """
    CSVファイルの1列目（Consumer列）から重複を除いた値のリストを返す。

    Parameters:
        file_path (str): CSVファイルのパス。

    Returns:
        list: 重複を除いたConsumer名のリスト。
    """
    unique_consumers = set()

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # ヘッダーをスキップ
        for row in reader:
            if row:  # 空行をスキップ
                unique_consumers.add(row[0])

    return list(unique_consumers)

def load_and_clean_csv(file_path):
    """CSVを読み込んで、日付列を処理し、NaT行を除外"""
    if not os.path.isfile(file_path):
        return None

    try:
        df = pd.read_csv(
            file_path,
            encoding='utf-8-sig',
            usecols=[0, 1, 2],
            header=0,
            names=["計測日", "計測時間", "全体"]
        )
        df["計測日"] = pd.to_datetime(df["計測日"], errors='coerce', format='%Y/%m/%d')
        df = df.dropna(subset=["計測日"])
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def is_valid_period(df, target_start, target_end):
    """対象期間すべてをカバーしているかチェック"""
    min_date = df["計測日"].min()
    max_date = df["計測日"].max()
    return min_date <= target_start and max_date >= target_end

def filter_target_dates(df, target_dates, expected_len=1464):
    """対象日のデータだけに絞り、期待する行数があるかを確認"""
    df_filtered = df[df["計測日"].isin(target_dates)]
    if len(df_filtered) != expected_len:
        return None
    return df_filtered

def make_pivot(df):
    """日毎のピボット（時間×日）を作成"""
    return df.pivot(index='計測時間', columns='計測日', values='全体')

def extract_consumer_name(file_name):
    return file_name.replace('.csv', '')

def is_complete_year_data(df, target_dates, expected_rows):
    """指定された日付範囲に対するデータが完全に揃っているか判定"""
    df_period = df[df["計測日"].isin(target_dates)]
    return df_period if len(df_period) == expected_rows else None

def calc_hourly_stats(pivot):
    """ピボットから時間ごとの平均と標準偏差を計算"""
    hourly_mean = pivot.mean(axis=1)
    hourly_std = pivot.std(axis=1, ddof=0)
    x = hourly_mean.index.astype(int).values
    y = hourly_mean.values
    yerr = hourly_std.values
    return x, y, yerr

def plot_hourly_stats(x, y, yerr, linestyle):
    """エラーバーつきの折れ線グラフを描画"""
    plt.plot(x, y, linestyle=linestyle)
    plt.fill_between(x, np.array(y) - np.array(yerr), np.array(y) + np.array(yerr), alpha=0.1)

def process_files(
    df_list,
    data_dir,
    target_start_md,
    target_end_md,
    expected_rows,
    target_days,
    consumer_profiles_by_contract,
    excluded_files,
    contract_type_key='契約電力',
    target_hour=None
):
    need_time_zfill = target_hour is not None  # target_hour指定時は必ずゼロ埋め

    for _, row in df_list.iterrows():
        file_name = row['ファイル名']
        consumer_name = extract_consumer_name(file_name)
        contract_type = row[contract_type_key]
        path = os.path.join(data_dir, file_name)

        df_raw = load_and_clean_csv(path)
        if df_raw is None:
            continue

        # 日付加工
        df_raw["計測日"] = pd.to_datetime(df_raw["計測日"], errors='coerce', format='%Y/%m/%d')
        df_raw = df_raw.dropna(subset=["計測日"])
        df_raw["年"] = df_raw["計測日"].dt.year
        df_raw["月日"] = df_raw["計測日"].dt.strftime('%m-%d')

        if need_time_zfill:
            df_raw["計測時間"] = df_raw["計測時間"].astype(str).str.zfill(2)

        file_valid = False

        for year in sorted(df_raw["年"].unique()):
            target_start = pd.to_datetime(f"{year}-{target_start_md}")
            target_end = pd.to_datetime(f"{year}-{target_end_md}")
            target_dates = pd.date_range(start=target_start, end=target_end)

            df_period = is_complete_year_data(df_raw, target_dates, expected_rows)
            if df_period is None:
                continue

            pivot = make_pivot(df_period)
            pivot.index = (
                pivot.index.astype(str)
                .str.extract(r'(\d{1,2})')[0]
                .astype(int)
                .astype(str)
                .str.zfill(2)
            )

            if pivot.shape[1] != target_days or pivot.isnull().values.any():
                continue

            if target_hour is not None:
                if target_hour not in pivot.index:
                    continue
                y = pivot.loc[target_hour].values
                consumer_profiles_by_contract[contract_type].append((y, consumer_name, year))
            else:
                consumer_profiles_by_contract[contract_type].append((pivot, consumer_name, year))

            file_valid = True

        if not file_valid:
            excluded_files.append(f"{file_name}（データ不足または対象期間なし）")
