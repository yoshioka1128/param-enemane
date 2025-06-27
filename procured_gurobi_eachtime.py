import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import os

# 入力ファイル
df_pred = pd.read_csv('output/predicted_negawatt_hourly_stats.csv')
df_target = pd.read_csv('output/procured_negawatt_hour11_21_ratio0_80.csv')

# 結果格納用
result_rows = []

# 各時間帯ごとに処理
for _, row in df_target.iterrows():
    hour = int(row['Hour'])
    target = row['Procured']

    # 対象時間帯データ抽出
    df_hour = df_pred[df_pred['Hour'] == hour]
    consumers = df_hour['Consumer'].tolist()
    means = df_hour['Mean'].tolist()

    # モデル作成
    m = gp.Model()
    m.Params.OutputFlag = 0  # ログ非表示

    # バイナリ変数：選ばれるかどうか
    x = m.addVars(len(consumers), vtype=GRB.BINARY, name="x")

    # 合計量
    total_mean = gp.quicksum(means[i] * x[i] for i in range(len(consumers)))

    # 目的関数：合計値とターゲットとの差の絶対値を最小化
    diff_pos = m.addVar(lb=0, name="diff_pos")
    diff_neg = m.addVar(lb=0, name="diff_neg")
    m.addConstr(total_mean - target == diff_pos - diff_neg)
    m.setObjective(diff_pos + diff_neg, GRB.MINIMIZE)

    # 解く
    m.optimize()

    # 解の抽出
    selected = [consumers[i] for i in range(len(consumers)) if x[i].X > 0.5]
    total = sum(means[i] for i in range(len(consumers)) if x[i].X > 0.5)
    diff = abs(total - target)

    total_consumers = len(consumers)
    selected_count = len(selected)
    print(f"Hour {hour}: 選択された需要家数 = {selected_count} / 全需要家数 = {total_consumers}")
    print(f"total {total}, target{target}, diff{diff}")

    result_rows.append({
        'Hour': hour,
        'Procured': target,
        'Selected_Consumers': ';'.join(selected),
        'Total_Mean': total,
        'Difference': diff
    })

# 出力保存
df_result = pd.DataFrame(result_rows)
os.makedirs('output', exist_ok=True)
df_result.to_csv('output/optimal_consumer_combination_eachtime.csv', index=False)


print("optimal_consumer_combination.csv を出力しました。")
