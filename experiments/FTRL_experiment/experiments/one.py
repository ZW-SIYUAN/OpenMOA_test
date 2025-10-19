"""test_news20_with_auc.py - α=0.2，计算 AUC (修复 deprecation warning)"""
import sys, os
sys.path.insert(0, os.path.abspath('./src'))

from pathlib import Path
import numpy as np
import time
from capymoa.stream import LibsvmStream
from capymoa.classifier import FTRLClassifier

DATA_DIR = Path("C:/reposity clone/OpenMOA/experiments/FTRL_experiment/data")

def calculate_auc(y_true, y_scores):
    """手动计算 AUC"""
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)
    
    tpr = tp / n_pos
    fpr = fp / n_neg
    
    # ✅ 使用 trapezoid 替代 trapz
    auc = np.trapezoid(tpr, fpr)
    return auc

print("="*80)
print("news20: α=0.2 (固定)，计算 AUC")
print("="*80)

file_path = DATA_DIR / 'news20.binary'

if not file_path.exists():
    print(f"❌ 文件不存在: {file_path}")
    exit(1)

print(f"\n论文结果: AUC = 98.9%, Sparsity = 5.2%\n")

alpha = 0.2
lambdas = [0.0005]

print(f"{'Lambda':<12} {'AUC':<12} {'Accuracy':<12} {'Sparsity':<12} {'Density':<12} {'Time(s)':<10}")
print("-"*90)

for lambda_val in lambdas:
    t_start = time.time()
    
    stream = LibsvmStream(
        path=file_path,
        dataset_name='news20',
        target_type='categorical'
    )
    
    learner = FTRLClassifier(
        schema=stream.get_schema(),
        alpha=alpha,
        beta=1.0,
        l1=lambda_val,
        l2=1.0
    )
    
    y_true = []
    y_scores = []
    correct = 0
    
    for instance in stream:
        # 获取预测概率
        proba = learner.predict_proba(instance)
        y_scores.append(proba[1])  # 正类概率
        
        # 预测类别
        pred = learner.predict(instance)
        if pred == instance.y_index:
            correct += 1
        
        y_true.append(instance.y_index)
        
        # 训练
        learner.train(instance)
    
    t_elapsed = time.time() - t_start
    
    # 计算指标
    accuracy = correct / len(stream) * 100
    auc = calculate_auc(y_true, y_scores) * 100
    sparsity = learner.get_sparsity()
    density = 1.0 - sparsity
    
    status = ""
    if abs(auc - 98.9) < 2.0 and abs(sparsity - 0.052) < 0.05:
        status = "✅ 接近论文"
    elif abs(sparsity - 0.052) < 0.05:
        status = "🎯 稀疏性匹配"
    
    print(f"{lambda_val:<12.6f} {auc:<12.3f} {accuracy:<12.3f} {sparsity:<12.3f} "
          f"{density:<12.3f} {t_elapsed:<10.1f} {status}")

print("\n" + "="*80)
print("目标: AUC ≈ 98.9%, Sparsity ≈ 5.2%")
print("="*80)