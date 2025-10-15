"""demo_hyperparameter_tuning.py - 超参数调优"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Fried
from capymoa.regressor import FESLRegressor
from capymoa.evaluation import prequential_evaluation
import itertools

stream = Fried()
schema = stream.schema

# 定义超参数搜索空间
param_grid = {
    'overlap_size': [50, 100, 200],
    'learning_rate_scale': [0.05, 0.1, 0.2],
    'ensemble_method': ['combination', 'selection']
}

# 生成所有组合
keys = param_grid.keys()
values = param_grid.values()
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Running {len(experiments)} experiments...\n")

best_mae = float('inf')
best_params = None

# 测试每个配置
for i, params in enumerate(experiments, 1):
    stream.restart()
    
    learner = FESLRegressor(
        schema=schema,
        s1_feature_indices=list(range(0, 6)),
        s2_feature_indices=list(range(4, 10)),
        switch_point=5000,
        random_seed=42,
        **params
    )
    
    results = prequential_evaluation(
        stream=stream,
        learner=learner,
        max_instances=5000,
        window_size=100
    )
    
    mae = results['cumulative'].mae()
    
    print(f"Exp {i}/{len(experiments)}: {params}")
    print(f"  MAE: {mae:.4f}\n")
    
    if mae < best_mae:
        best_mae = mae
        best_params = params

print("="*50)
print("Best Configuration:")
print(f"  Parameters: {best_params}")
print(f"  MAE: {best_mae:.4f}")
print("="*50)