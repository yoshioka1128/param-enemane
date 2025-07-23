import pandas as pd
import os

# CSV読み込み
df_list = pd.read_csv('list_60.csv', encoding='cp932')
df_list.columns = df_list.columns.str.strip()

file_names = df_list['ファイル名']
contract_power = df_list['契約電力']

# チェック対象ディレクトリ
dirs = ['raw']

# ファイルごとの結果リスト
results = []

# 必要な列名だけ指定
use_cols = ["計測日", "計測時間", "全体"]

for file_name, power in zip(file_names, contract_power):
    found = False
    file_path = None
    for d in dirs:
        temp_path = os.path.join(d, file_name)
        if os.path.isfile(temp_path):
            found = True
            file_path = temp_path
            break

    # 存在しない場合でも記録は行う
    file_result = {
        'file': file_name,
        'exists': found,
        '契約電力': power,
        'start_date': None,
        'end_date': None,
        'date_range': None,
    }

    # ファイルが存在する場合のみ日付抽出
    if found:
        try:
            df = pd.read_csv(
                file_path,
                encoding='utf-8-sig',
                usecols=[0, 1, 2],
                header=0,
                names=["計測日", "計測時間", "全体"]
            )
#            df = pd.read_csv(file_path, usecols=use_cols, encoding='utf-8-sig')
            df.columns = df.columns.str.strip()
        
            df['計測日'] = pd.to_datetime(df['計測日'], errors='coerce', format='%Y/%m/%d')
            valid_dates = df['計測日'].dropna()

            if valid_dates.empty:
                print(f"[INFO] 計測日がすべてNaTでした: {file_name}")
                print(df['計測日'].head())  # データの中身確認
                exit()

            if not valid_dates.empty:
                start = valid_dates.min().strftime('%Y/%m/%d')
                end = valid_dates.max().strftime('%Y/%m/%d')
                file_result['start_date'] = start
                file_result['end_date'] = end
                file_result['date_range'] = f"{start}～{end}"
        except Exception as e:
            print(f"[ERROR] {file_name}: {e}")
            
    results.append(file_result)

# DataFrame化
df_results = pd.DataFrame(results)
missing_files = df_results[~df_results['exists']]

# --- 元の出力 ---
print(f"総ファイル数: {len(df_results)}")
print(f"存在するファイル数: {df_results['exists'].sum()}")
print(f"存在しないファイル数: {len(missing_files)}")

# --- 計測期間で分類して、その中で契約電力ごとの件数を集計 ---
print("\n計測期間ごとの契約電力別ファイル数:")

grouped = df_results[df_results['exists']].groupby(['date_range', '契約電力'])

# 結果を整理して出力
output = grouped.size().unstack(fill_value=0)

count_sum = 0
for date_range in output.index:
    print(f"\n 計測期間: {date_range}")
    for power_type in output.columns:
        count = output.loc[date_range, power_type]
        print(f"  {power_type}: {count} 件")
        count_sum += count
    if date_range == None:
        print('date_range')
print('count sum: ', count_sum)

# --- 追加: 契約電力ごとの合計ファイル数（date_range 無視） ---
print("\n契約電力ごとの合計ファイル数（全期間合計）:")
power_totals = df_results[df_results['exists']].groupby('契約電力').size()
for power_type, count in power_totals.items():
    print(f"  {power_type}: {count} 件")

# データフレーム化
df_results = pd.DataFrame(results)

# 存在するファイルだけフィルタ
df_existing = df_results[df_results['exists']]

# 日付範囲ごとの契約電力別の件数を集計
date_power_counts = {}
for _, row in df_existing.iterrows():
    date_range = row['date_range']
    power = row['契約電力']
    if pd.isna(date_range):
        continue
    if date_range not in date_power_counts:
        date_power_counts[date_range] = {}
    if power not in date_power_counts[date_range]:
        date_power_counts[date_range][power] = 0
    date_power_counts[date_range][power] += 1

# ===  チェック ===
categorized_total = sum(
    sum(power_counts.values())
    for power_counts in date_power_counts.values()
)
existing_files_count = df_existing.shape[0]

print("\n ファイル分類集計チェック:")
print(f"分類されたファイル数合計: {categorized_total}")
print(f"存在するファイル数: {existing_files_count}")

if categorized_total == existing_files_count:
    print(" 分類されたファイル数と存在するファイル数は一致しています。")
else:
    print("️ 不一致があります。データに漏れや重複がないか確認してください。")

# date_range または 契約電力 が欠損している行を抽出
df_unclassified = df_existing[df_existing['date_range'].isna() | df_existing['契約電力'].isna()]

print("\n 計測開始日が1970年のファイル一覧:")
df_1970 = df_results[(df_results['start_date'].str.startswith('1970')) & (df_results['exists'])]
for fname in df_1970['file']:
    print(fname)        

