"""demo_export_results.py - 手动导出详细结果"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

import pandas as pd
import json
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

# 运行评估
results = prequential_evaluation(
    stream=stream,
    learner=learner,
    max_instances=10000,
    window_size=100,
    store_predictions=True,
    store_y=True,
    progress_bar=True
)

# 创建导出目录
output_dir = '.'
os.makedirs(output_dir, exist_ok=True)

print("\nExporting results...")

# 1. 导出累积结果
cumulative_metrics = results['cumulative'].metrics_dict()
cumulative_df = pd.DataFrame([cumulative_metrics])
cumulative_df.to_csv(f'{output_dir}/cumulative_results.csv', index=False)

# 2. 导出窗口结果
windowed_df = results['windowed'].metrics_per_window()
windowed_df.to_csv(f'{output_dir}/windowed_results.csv', index=False)

# 3. 导出预测结果
predictions_df = pd.DataFrame({
    'ground_truth': results._ground_truth_y,
    'prediction': results._predictions,
    'error': [p - y for p, y in zip(results._predictions, results._ground_truth_y)]
})
predictions_df.to_csv(f'{output_dir}/predictions.csv', index=False)

# 4. 导出摘要统计
summary = {
    'Algorithm': 'FESLRegressor',
    'MAE': results['cumulative'].mae(),
    'RMSE': results['cumulative'].rmse(),
    'R2': results['cumulative'].r2(),
    'Total_Instances': 10000,
    'Wallclock_Time_sec': results.wallclock,
    'CPU_Time_sec': results.cpu_time
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv(f'{output_dir}/summary.csv', index=False)

# 5. 导出配置信息（JSON格式）
config = {
    'learner': str(learner),
    'dataset': 'Fried',
    's1_indices': list(range(0, 6)),
    's2_indices': list(range(4, 10)),
    'overlap_size': 100,
    'switch_point': 5000,
    'learning_rate_scale': 0.1,
    'random_seed': 42
}
with open(f'{output_dir}/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"\nResults exported to '{output_dir}/':")
print("  ✓ cumulative_results.csv")
print("  ✓ windowed_results.csv")
print("  ✓ predictions.csv")
print("  ✓ summary.csv")
print("  ✓ config.json")