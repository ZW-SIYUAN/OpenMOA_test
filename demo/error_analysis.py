"""demo_error_analysis.py - 预测误差分析"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

import numpy as np
import matplotlib.pyplot as plt
from capymoa.datasets import Fried
from capymoa.regressor import FESLRegressor
from capymoa.evaluation import prequential_evaluation

stream = Fried()
schema = stream.schema

learner = FESLRegressor(
    schema=schema,
    s1_feature_indices=list(range(0, 6)),
    s2_feature_indices=list(range(4, 10)),
    overlap_size=100,
    switch_point=5000,
    learning_rate_scale=0.1,
    random_seed=42
)

# 保存预测值和真实值
results = prequential_evaluation(
    stream=stream,
    learner=learner,
    max_instances=10000,
    window_size=100,
    store_predictions=True,
    store_y=True,
    progress_bar=True
)

# 使用正确的属性访问（带下划线）
predictions = np.array(results._predictions)
ground_truth = np.array(results._ground_truth_y)
errors = predictions - ground_truth

# 统计分析
print("\n" + "="*50)
print("Error Analysis")
print("="*50)
print(f"Mean Error: {np.mean(errors):.4f}")
print(f"Std Error: {np.std(errors):.4f}")
print(f"Max Error: {np.max(np.abs(errors)):.4f}")
print(f"Median Error: {np.median(errors):.4f}")
print("="*50)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 误差分布直方图
axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0].set_title('Error Distribution', fontweight='bold')
axes[0].set_xlabel('Prediction Error')
axes[0].set_ylabel('Frequency')

# 预测 vs 真实值散点图
axes[1].scatter(ground_truth, predictions, alpha=0.5, s=10)
axes[1].plot([ground_truth.min(), ground_truth.max()], 
             [ground_truth.min(), ground_truth.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_title('Predictions vs Ground Truth', fontweight='bold')
axes[1].set_xlabel('Ground Truth')
axes[1].set_ylabel('Predictions')
axes[1].legend()

# 误差随时间变化
axes[2].plot(np.abs(errors), alpha=0.6, linewidth=0.5)
axes[2].axvline(x=5000, color='red', linestyle='--', label='Switch Point')
axes[2].set_title('Absolute Error over Time', fontweight='bold')
axes[2].set_xlabel('Instance')
axes[2].set_ylabel('Absolute Error')
axes[2].legend()

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=300)
print("\nError analysis saved to 'error_analysis.png'")
plt.show()