import pandas as pd
import os

# CSV読み込み
df_list = pd.read_csv('list_30.csv', encoding='cp932')

# 念のため空白除去
df_list.columns = df_list.columns.str.strip()

file_names = df_list['ファイル名']

# raw, raw2 ディレクトリ
dirs = ['raw', 'raw2']

results = []

for file_name in file_names:
    found = False
    for d in dirs:
        file_path = os.path.join(d, file_name)
        if os.path.isfile(file_path):
            found = True
            break
    results.append({
        'file': file_name,
        'exists': found
    })

df_results = pd.DataFrame(results)
missing_files = df_results[~df_results['exists']]

print(f"総ファイル数: {len(df_results)}")
print(f"存在するファイル数: {df_results['exists'].sum()}")
print(f"存在しないファイル数: {len(missing_files)}")

# 必要に応じて：print(missing_files)

# 各地方（所在地）ごとのファイル件数を出力
area_counts = df_list['所在地'].value_counts()

print("\n各地方のファイル件数:")
for area, count in area_counts.items():
    print(f"{area}: {count} 件")
