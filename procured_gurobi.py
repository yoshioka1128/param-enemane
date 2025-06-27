import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# 入力ファイル
df_pred = pd.read_csv('output/predicted_negawatt_hourly_stats.csv')
df_target = pd.read_csv('output/procured_negawatt_hour15_20_ratio0_80.csv')

# 結果格納用
result_rows = []

# 時間帯の一覧（3の倍数）
group_start_hours = sorted(df_target['Hour'].unique())
group_start_hours = [h for h in group_start_hours if h % 3 == 0]

for start_hour in group_start_hours:
    group_hours = [start_hour, start_hour + 1, start_hour + 2]

    # 対象となる3時間すべてのデータを抽出
    df_group = df_pred[df_pred['Hour'].isin(group_hours)]
    df_targets_group = df_target[df_target['Hour'].isin(group_hours)]

    # すべての需要家（全時間帯共通で同じconsumerにする）
    consumers = sorted(df_group['Consumer'].unique())
    n = len(consumers)

    # 各時間帯ごとのmean値を辞書にまとめる
    means_by_hour = {
        h: df_group[df_group['Hour'] == h].set_index('Consumer')['Mean'].reindex(consumers).fillna(0).tolist()
        for h in group_hours
    }
    targets_by_hour = {
        row['Hour']: row['Procured']
        for _, row in df_targets_group.iterrows()
    }

    # モデル作成
    m = gp.Model()
    m.Params.OutputFlag = 0  # ログ非表示

    # バイナリ変数（全時間帯共通）
    x = m.addVars(n, vtype=GRB.BINARY, name="x")

    # 各時間帯の誤差変数
    diffs_pos = {}
    diffs_neg = {}

    for h in group_hours:
        means = means_by_hour[h]
        target = targets_by_hour[h]

        total_mean = gp.quicksum(means[i] * x[i] for i in range(n))

        diff_pos = m.addVar(lb=0, name=f"diff_pos_{h}")
        diff_neg = m.addVar(lb=0, name=f"diff_neg_{h}")
        m.addConstr(total_mean - target == diff_pos - diff_neg)

        diffs_pos[h] = diff_pos
        diffs_neg[h] = diff_neg

    # 目的関数：誤差の合計最小化
    m.setObjective(
        gp.quicksum(diffs_pos[h] + diffs_neg[h] for h in group_hours),
        GRB.MINIMIZE
    )

    # 最適化
    m.optimize()

    # 解の抽出
    selected = [consumers[i] for i in range(n) if x[i].X > 0.5]
    selected_count = len(selected)

    for h in group_hours:
        means = means_by_hour[h]
        target = targets_by_hour[h]
        total = sum(means[i] for i in range(n) if x[i].X > 0.5)
        diff = abs(total - target)
        print(f"Hour {h}: total={total:.2f}, target={target:.2f}, diff={diff:.2f}, 選択数={selected_count}/{n}")

        result_rows.append({
            'Hour': h,
            'Procured': target,
            'Selected_Consumers': ';'.join(selected),
            'Total_Mean': total,
            'Difference': diff
        })

# 出力保存
df_result = pd.DataFrame(result_rows)
os.makedirs('output', exist_ok=True)
def_result.to_csv('output/optimal_consumer_combination_3step.csv', index=False)


print("optimal_consumer_combination.csv を出力しました。")
