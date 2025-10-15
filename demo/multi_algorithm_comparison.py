"""demo_algorithm_comparison.py - 多算法性能对比"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Fried
from capymoa.regressor import FESLRegressor, SGDRegressor, PassiveAggressiveRegressor
from capymoa.evaluation import prequential_evaluation_multiple_learners

# 准备数据流
stream = Fried()
schema = stream.schema

# 配置多个学习器
learners = {
    'FESL': FESLRegressor(
        schema=schema,
        s1_feature_indices=list(range(0, 6)),
        s2_feature_indices=list(range(4, 10)),
        overlap_size=100,
        switch_point=5000,
        learning_rate_scale=0.1,
        random_seed=42
    ),
    'SGD': SGDRegressor(schema=schema, random_seed=42),
    'PA': PassiveAggressiveRegressor(schema=schema, random_seed=42)
}

# 运行对比评估
print("Running multi-learner comparison...")
results = prequential_evaluation_multiple_learners(
    stream=stream,
    learners=learners,
    max_instances=10000,
    window_size=100,
    progress_bar=True
)

# 打印对比结果
print("\n" + "="*50)
print("Algorithm Comparison Results")
print("="*50)
for name, result in results.items():
    mae = result['cumulative'].mae()
    rmse = result['cumulative'].rmse()
    print(f"{name:12s} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
print("="*50)