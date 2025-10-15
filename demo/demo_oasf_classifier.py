"""demo_oasf_evolving.py"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

import matplotlib.pyplot as plt
from capymoa.datasets import Electricity
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import OASFClassifier
from capymoa.stream import EvolvingFeatureStream  # 使用 capymoa 前缀

# 创建演化特征流
base_stream = Electricity()
evolving_stream = EvolvingFeatureStream(
    base_stream=base_stream,
    d_min=2,
    d_max=6,
    evolution_pattern="pyramid",
    total_instances=10000,
    feature_selection="prefix",
    random_seed=42
)

# 创建 OASF 学习器
oasf_learner = OASFClassifier(
    schema=evolving_stream.get_schema(),
    lambda_param=0.01,
    mu=10,
    L=100,
    d_max=10,
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

print(f"\nAccuracy: {results['cumulative'].accuracy():.2f}%")
print(f"Kappa: {results['cumulative'].kappa():.4f}")