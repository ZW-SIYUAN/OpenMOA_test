"""OSLMF Classifier Demo - 简单版"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))
from capymoa.datasets import Electricity
from capymoa.stream import CapriciousStream
from capymoa.classifier import OSLMFClassifier
from tqdm import tqdm
import random

print("=" * 70)
print("OSLMF Classifier Demo - Capricious Stream (50% labels)")
print("=" * 70)

# 创建数据流
base_stream = Electricity()
stream = CapriciousStream(
    base_stream=base_stream,
    missing_ratio=0.05,
    total_instances=3000,
    min_features=2,
    random_seed=42
)

print(f"\n[Stream Info]")
print(f"  Features: {base_stream.get_schema().get_num_attributes()}")
print(f"  Missing ratio: 50%")
print(f"  Classes: 2")

# 创建分类器
learner = OSLMFClassifier(
    schema=stream.get_schema(),
    window_size=200,
    buffer_size=100,
    learning_rate=0.01,
    random_seed=42
)

print(f"\n[Classifier]")
print(f"  Window: 200, Buffer: 100")

# 评估
print(f"\n[Running Evaluation...]")
print("-" * 70)

random.seed(42)
correct = 0
total = 0
labeled_count = 0

stream.restart()
for _ in tqdm(range(3000), desc="Processing", ncols=70):
    instance = stream.next_instance()
    
    # 预测
    prediction = learner.predict(instance)
    if prediction == instance.y_index:
        correct += 1
    total += 1
    
    # 训练（50% 有标签）
    has_label = random.random() < 0.5
    instance._has_true_label = has_label
    if has_label:
        labeled_count += 1
    
    learner.train(instance)

print("-" * 70)

# 结果
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\n[Performance]")
print(f"  Accuracy: {correct/total*100:.4f}%")
print(f"  Labeled: {labeled_count}/{total} ({labeled_count/total*100:.1f}%)")
print(f"  Ensemble α: {learner.ensemble_weight:.4f}")
print(f"\n[OSLMF Info]")
print(f"  Updates: {learner._num_updates}")
print(f"  Loss (obs/lat): {learner._loss_observed:.1f} / {learner._loss_latent:.1f}")

print("\n" + "=" * 70)
print("✓ Demo completed")
print("=" * 70)