import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

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

