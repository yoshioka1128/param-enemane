#!/bin/bash

# 実行したい時間帯（1〜24）を生成
hours=$(seq 1 24)

# 各時間帯で Python スクリプトを実行
for hour in $hours; do
    echo "時間帯 $hour の共分散行列を計算中..."
    echo "$hour" | python covariance_matrix_mixup_restricted.py
done
