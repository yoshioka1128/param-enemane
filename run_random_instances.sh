#!/bin/bash

# 実行する L のリスト
sizes=(210 105 60 30 18)

# 順番に実行
for L in "${sizes[@]}"; do
    echo "Running with L=$L"
    # L を入力 → seed はデフォルト
    echo -e "$L\n" | python3 read_files_random_instance.py
done
