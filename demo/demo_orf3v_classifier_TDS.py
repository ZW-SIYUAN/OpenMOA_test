"""demo_orf3v_classifier.py - ORF3V Classifier Demo"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Electricity
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import ORF3VClassifier
from capymoa.stream import EvolvingFeatureStream

# TDS 模式（推荐）
stream = EvolvingFeatureStream(
    base_stream=Electricity(),
    evolution_pattern="tds",
    d_max=6,
    total_instances=10000,
    random_seed=42
)

# 创建分类器
learner = ORF3VClassifier(
    schema=stream.get_schema(),
    n_stumps=20,     # 决策树桩的数量
    alpha=0.3,  # 权重更新时的学习率
    grace_period=100,   #在前 100 个样本时只收集统计，不建立森林
    replacement_interval=50,  # 每 50 个样本替换一次最老的树
    random_seed=42  # 随机种子
)

print(f"Evaluating {learner} on EvolvingFeatureStream (TDS mode)")
print(f"Dataset: Electricity")
print("-" * 60)

# 评估
results = prequential_evaluation(
    stream=stream,
    learner=learner,
    max_instances=10000,
    window_size=100
)


print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Accuracy:              {results['cumulative'].accuracy():.2f}%")
print(f"Kappa:                 {results['cumulative'].kappa():.4f}")
print(f"Kappa Temporal:        {results['cumulative'].kappa_t():.4f}")
print(f"Kappa M:               {results['cumulative'].kappa_m():.4f}")
print(f"F1 Score:              {results['cumulative'].f1_score():.2f}%")
print(f"Precision:             {results['cumulative'].precision():.2f}%")
print(f"Recall:                {results['cumulative'].recall():.2f}%")
print(f"Instances processed:   {results['cumulative'].instances_seen}")  
