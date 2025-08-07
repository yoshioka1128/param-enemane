#!/bin/bash

# 実行したい時間帯のリスト
hours=(1 2 3 4 5 6 7 8 9)

# 各時間帯でスクリプトを実行
for hour in "${hours[@]}"; do
    echo "時間帯 $hour の共分散行列を計算中..."
    echo "$hour" | python covariance_matrix_mixup_restricted.py
done
