"""demo_rsol_classifier.py - RSOL Classifier 演示"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Electricity
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import RSOLClassifier
from capymoa.stream import EvolvingFeatureStream

# 创建基础分类数据流
base_stream = Electricity()
print(f"Electricity 特征数: {base_stream.get_schema().get_num_attributes()}")

# 包装为特征演化流
evolving_stream = EvolvingFeatureStream(
    base_stream=base_stream,
    d_min=2,
    d_max=6,
    evolution_pattern="pyramid",
    total_instances=10000,
    feature_selection="prefix",
    random_seed=42
)

# 创建 RSOL 分类器
rsol_learner = RSOLClassifier(
    schema=evolving_stream.get_schema(),
    lambda_param=0.01,    # RSOL 论文推荐值（更激进的稀疏化）
    mu=10,               # RSOL 论文推荐值
    L=100,             # 更长的滑动窗口
    d_max=10,
    random_seed=42
)

print(f"\n学习器: {rsol_learner}")

# 运行评估
results = prequential_evaluation(
    stream=evolving_stream,
    learner=rsol_learner,
    max_instances=10000,
    window_size=100,
    progress_bar=True
)

# 打印结果
print(f"\nAccuracy: {results['cumulative'].accuracy():.2f}%")
print(f"Kappa: {results['cumulative'].kappa():.4f}")

# 在 demo 中添加
print(f"Non-zero weights: {np.count_nonzero(rsol_learner.W)}")
print(f"Total weights: {rsol_learner.W.size}")
print(f"Sparsity: {1 - np.count_nonzero(rsol_learner.W) / rsol_learner.W.size:.2%}")