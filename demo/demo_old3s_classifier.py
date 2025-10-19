"""Demo script for OLD³S Classifier"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Electricity
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import OLD3SClassifier

# Load stream
electricity_stream = Electricity()
schema = electricity_stream.schema

print(f"Number of features: {schema.get_num_attributes()}")

# Create OLD³S learner
old3s_learner = OLD3SClassifier(
    schema=schema,
    s1_feature_indices=[0,1,2,3],   # 前 4 个特征
    s2_feature_indices=[2,3,4,5],  # 后 29 个特征（25-53）
    overlap_size=500,                      # 重叠期大小
    switch_point=5000,                     # 切换点
    latent_dim=20,                         # VAE 潜在空间维度
    hidden_dim=128,                        # VAE 隐藏层维度
    num_hbp_layers=5,                      # HBP 分类器层数
    learning_rate=0.001,                   # 学习率
    beta=0.9,                              # HBP 权重衰减率
    eta=-0.05,                             # 集成权重参数
    random_seed=42
)

# Run evaluation with progress bar
results = prequential_evaluation(
    stream=electricity_stream,
    learner=old3s_learner,
    max_instances=10000,
    window_size=500,
    progress_bar=True
)

# Print final results
print(f"\nFinal Accuracy: {results['cumulative'].accuracy():.2f}%")
print(f"Kappa: {results['cumulative'].kappa():.4f}")