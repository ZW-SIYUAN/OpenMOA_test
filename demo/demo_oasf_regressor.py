"""demo_oasf_regressor.py - OASF Regressor 演示"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Fried
from capymoa.evaluation import prequential_evaluation
from capymoa.regressor import OASFRegressor
from capymoa.stream import EvolvingFeatureStream

# 创建基础回归数据流
base_stream = Fried()
print(f"Fried 特征数: {base_stream.get_schema().get_num_attributes()}")

# 包装为特征演化流
evolving_stream = EvolvingFeatureStream(
    base_stream=base_stream,
    d_min=3,
    d_max=10,
    evolution_pattern="pyramid",
    total_instances=10000,
    feature_selection="prefix",
    random_seed=42
)

# 创建 OASF 回归器
oasf_learner = OASFRegressor(
    schema=evolving_stream.get_schema(),
    lambda_param=0.01,
    mu=10,
    L=100,
    d_max=15,
    random_seed=42
)

# 运行评估
results = prequential_evaluation(
    stream=evolving_stream,
    learner=oasf_learner,
    max_instances=10000,
    window_size=100,
    progress_bar=True
)

# 打印结果（使用正确的方法名）
print(f"\nMAE: {results['cumulative'].mae():.4f}")
print(f"RMSE: {results['cumulative'].rmse():.4f}")
print(f"R²: {results['cumulative'].r2():.4f}")