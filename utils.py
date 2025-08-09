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

def plot_hourly_stats(x, y, yerr):
    """エラーバーつきの折れ線グラフを描画"""
    plt.plot(x, y)
    plt.fill_between(x, np.array(y) - np.array(yerr), np.array(y) + np.array(yerr), alpha=0.1)
