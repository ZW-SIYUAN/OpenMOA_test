#!/usr/bin/env python3
"""
运行 FESLClassifier 在指定数据集的 10 个 perm 文件上，
计算准确率、均值、标准差并保存结果。

使用方法：
  直接修改下面的 dataset_name 变量即可。
"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

import numpy as np
from pathlib import Path
from capymoa.stream import ARFFStream
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import FESLClassifier

# ===============================
dataset_name = "credit-a"  
# ===============================
ensemble_method = "combination"     # "selection" 或 "combination"
BASE_DIR = Path("C:/reposity clone/OpenMOA/experiments/FESL_experiment/datasets/arff")
N_PERM = 10
N_SAMPLES = 653
SWITCH_POINT = N_SAMPLES // 2


def run_single_experiment(arff_path: Path) -> float:
    """在单个ARFF文件上运行FESLClassifier并返回accuracy（百分比）"""
    stream = ARFFStream(str(arff_path))

    fesl = FESLClassifier(
        schema=stream.schema,
        s1_feature_indices=list(range(15)),
        s2_feature_indices=list(range(15, 25)),
        overlap_size=10,
        switch_point=SWITCH_POINT,
        ensemble_method='combination',
        learning_rate_scale=1.0,
        random_seed=None
    )

    results = prequential_evaluation(
        stream, fesl,
        max_instances=N_SAMPLES,
        window_size=1,
        progress_bar=True
    )

    acc = results['cumulative'].accuracy()
    print(f"✅ {arff_path.name}: Accuracy = {acc:.3f}%")
    return acc


def main():
    print(f"\n📘 正在运行数据集: {dataset_name}")

    results_dir = Path("./experiments/FESL_experiment/results/"+ensemble_method)
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"{dataset_name}_fesl_results.txt"

    accuracies = []
    for i in range(1, N_PERM + 1):
        arff_path = BASE_DIR / f"{dataset_name}_perm_{i}.arff"
        if not arff_path.exists():
            print(f"⚠️ 未找到 {arff_path.name}，跳过。")
            continue

        acc = run_single_experiment(arff_path)
        accuracies.append(acc)

    if accuracies:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)  # 样本标准差

        print("\n📊 结果汇总：")
        print(f"Accuracies: {', '.join(f'{a:.3f}' for a in accuracies)}")
        print(f"Mean: {mean_acc:.3f}%")
        print(f"Std: {std_acc:.3f}%")

        with open(results_file, "w") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Accuracies: {', '.join(f'{a:.3f}' for a in accuracies)}\n")
            f.write(f"Mean: {mean_acc:.3f}%\n")
            f.write(f"Std: {std_acc:.3f}%\n")

        print(f"\n💾 结果已保存到 {results_file.resolve()}")
    else:
        print("❌ 没有成功运行的实验。")


if __name__ == "__main__":
    main()
