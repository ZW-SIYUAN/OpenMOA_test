"""demo_performance_visualization.py - 性能可视化"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Fried
from capymoa.regressor import FESLRegressor
from capymoa.evaluation import prequential_evaluation
import matplotlib.pyplot as plt

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

# 运行评估
results = prequential_evaluation(
    stream=stream,
    learner=learner,
    max_instances=10000,
    window_size=100,
    progress_bar=True
)

# 获取窗口结果
windowed_df = results['windowed'].metrics_per_window()

# 创建可视化
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# MAE 随时间变化
axes[0].plot(windowed_df['mae'], marker='o', linewidth=2)
axes[0].axvline(x=50, color='red', linestyle='--', label='Switch Point')
axes[0].set_title('MAE over Time', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Window')
axes[0].set_ylabel('MAE')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# RMSE 随时间变化
axes[1].plot(windowed_df['rmse'], marker='s', linewidth=2, color='orange')
axes[1].axvline(x=50, color='red', linestyle='--', label='Switch Point')
axes[1].set_title('RMSE over Time', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Window')
axes[1].set_ylabel('RMSE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fesl_performance.png', dpi=300)
print("\nVisualization saved to 'fesl_performance.png'")
plt.show()