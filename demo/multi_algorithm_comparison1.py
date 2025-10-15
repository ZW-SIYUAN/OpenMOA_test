"""demo_oasf_vs_rsol.py - 公平对比 OASF 和 RSOL 在特征演化场景下的表现"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from capymoa.datasets import Electricity
from capymoa.evaluation import prequential_evaluation_multiple_learners
from capymoa.classifier import OASFClassifier, RSOLClassifier
from capymoa.stream import EvolvingFeatureStream

print("="*70)
print("OASF vs RSOL: Feature Evolution Scenario Comparison")
print("="*70)

# 创建基础数据流
base_stream = Electricity()
original_features = base_stream.get_schema().get_num_attributes()
print(f"\nDataset: Electricity")
print(f"Original features: {original_features}")
print(f"Evaluation instances: 10,000")
print(f"Evolution pattern: Pyramid (2 → {original_features} → 2)")

# 包装为特征演化流
evolving_stream = EvolvingFeatureStream(
    base_stream=base_stream,
    d_min=2,
    d_max=original_features,
    evolution_pattern="pyramid",
    total_instances=10000,
    feature_selection="prefix",
    random_seed=42
)

print(f"\n{'='*70}")
print("Initializing Learners...")
print(f"{'='*70}")

# 定义两个学习器
learners = {
    'OASF': OASFClassifier(
        schema=evolving_stream.get_schema(),
        lambda_param=0.01,
        mu=10,
        L=100,
        d_max=20,
        random_seed=42
    ),
    'RSOL': RSOLClassifier(
        schema=evolving_stream.get_schema(),
        lambda_param=0.5,
        mu=10,
        L=100,
        d_max=20,
        random_seed=42
    )
}

for name, learner in learners.items():
    print(f"  {name}: {learner}")

print(f"\n{'='*70}")
print("Running Prequential Evaluation...")
print(f"{'='*70}\n")

# 运行评估
results = prequential_evaluation_multiple_learners(
    stream=evolving_stream,
    learners=learners,
    max_instances=10000,
    window_size=100,
    progress_bar=True
)

# 收集结果
print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}\n")

comparison_data = []
for name, result in results.items():
    learner_obj = learners[name]
    
    # 计算稀疏度
    non_zero = np.count_nonzero(learner_obj.W)
    total = learner_obj.W.size
    sparsity = (1 - non_zero / total) * 100
    
    # 兼容处理 wallclock 和 cpu_time（可能是方法或属性）
    try:
        wallclock_time = result.wallclock if not callable(result.wallclock) else result.wallclock()
        cpu_time_val = result.cpu_time if not callable(result.cpu_time) else result.cpu_time()
    except:
        wallclock_time = 0
        cpu_time_val = 0
    
    comparison_data.append({
        'Algorithm': name,
        'Accuracy (%)': result['cumulative'].accuracy(),
        'Kappa': result['cumulative'].kappa(),
        'Kappa-T': result['cumulative'].kappa_t(),
        'Precision (%)': result['cumulative'].precision(),
        'Recall (%)': result['cumulative'].recall(),
        'F1-Score (%)': result['cumulative'].f1_score(),
        'Wallclock (s)': wallclock_time,
        'CPU Time (s)': cpu_time_val,
        'Non-zero Weights': non_zero,
        'Total Weights': total,
        'Sparsity (%)': sparsity
    })

# 创建对比表格
df = pd.DataFrame(comparison_data)

# 显示主要指标
print("Performance Metrics:")
print("-" * 70)
metrics_df = df[['Algorithm', 'Accuracy (%)', 'Kappa', 'F1-Score (%)', 
                 'Wallclock (s)', 'Sparsity (%)']].copy()
print(metrics_df.to_string(index=False, float_format='%.4f'))

print(f"\n{'='*70}")
print("Detailed Analysis:")
print("-" * 70)

# 性能对比
acc_diff = abs(df.loc[0, 'Accuracy (%)'] - df.loc[1, 'Accuracy (%)'])
winner_acc = df.loc[df['Accuracy (%)'].idxmax(), 'Algorithm']
print(f"Accuracy difference: {acc_diff:.2f}% ({winner_acc} is better)")

kappa_diff = abs(df.loc[0, 'Kappa'] - df.loc[1, 'Kappa'])
winner_kappa = df.loc[df['Kappa'].idxmax(), 'Algorithm']
print(f"Kappa difference: {kappa_diff:.4f} ({winner_kappa} is better)")

# 时间对比
if df['Wallclock (s)'].iloc[0] > 0 and df['Wallclock (s)'].iloc[1] > 0:
    time_diff = abs(df.loc[0, 'Wallclock (s)'] - df.loc[1, 'Wallclock (s)'])
    faster = df.loc[df['Wallclock (s)'].idxmin(), 'Algorithm']
    speedup = time_diff / max(df['Wallclock (s)']) * 100
    print(f"Runtime difference: {time_diff:.4f}s ({faster} is {speedup:.1f}% faster)")

# 稀疏度对比
sparse_diff = abs(df.loc[0, 'Sparsity (%)'] - df.loc[1, 'Sparsity (%)'])
winner_sparse = df.loc[df['Sparsity (%)'].idxmax(), 'Algorithm']
print(f"Sparsity difference: {sparse_diff:.2f}% ({winner_sparse} is more sparse)")

print(f"\n{'='*70}")
print("Winner Summary:")
print("-" * 70)
print(f"  Best Accuracy:  {df.loc[df['Accuracy (%)'].idxmax(), 'Algorithm']} "
      f"({df['Accuracy (%)'].max():.2f}%)")
print(f"  Best Kappa:     {df.loc[df['Kappa'].idxmax(), 'Algorithm']} "
      f"({df['Kappa'].max():.4f})")
if df['Wallclock (s)'].min() > 0:
    print(f"  Fastest:        {df.loc[df['Wallclock (s)'].idxmin(), 'Algorithm']} "
          f"({df['Wallclock (s)'].min():.4f}s)")
print(f"  Most Sparse:    {df.loc[df['Sparsity (%)'].idxmax(), 'Algorithm']} "
      f"({df['Sparsity (%)'].max():.2f}%)")

print(f"\n{'='*70}\n")

# ========== 可视化 ==========
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

algorithms = df['Algorithm']
colors = ['#ff7f0e', '#2ca02c']

# 1. 准确率对比
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(algorithms, df['Accuracy (%)'], color=colors, alpha=0.8, edgecolor='black')
ax1.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=10)
ax1.set_ylim([75, 90])
ax1.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. Kappa 对比
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(algorithms, df['Kappa'], color=colors, alpha=0.8, edgecolor='black')
ax2.set_title('Kappa Statistic', fontsize=12, fontweight='bold')
ax2.set_ylabel('Kappa', fontsize=10)
ax2.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. F1-Score 对比
ax3 = fig.add_subplot(gs[0, 2])
bars = ax3.bar(algorithms, df['F1-Score (%)'], color=colors, alpha=0.8, edgecolor='black')
ax3.set_title('F1-Score', fontsize=12, fontweight='bold')
ax3.set_ylabel('F1-Score (%)', fontsize=10)
ax3.set_ylim([75, 90])
ax3.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. 运行时间对比
ax4 = fig.add_subplot(gs[1, 0])
if df['Wallclock (s)'].min() > 0:
    bars = ax4.bar(algorithms, df['Wallclock (s)'], color=colors, alpha=0.8, edgecolor='black')
    ax4.set_title('Runtime (Wallclock)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Time (seconds)', fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
else:
    ax4.text(0.5, 0.5, 'Timing data unavailable', ha='center', va='center', 
             transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Runtime (Wallclock)', fontsize=12, fontweight='bold')

# 5. 稀疏度对比
ax5 = fig.add_subplot(gs[1, 1])
bars = ax5.bar(algorithms, df['Sparsity (%)'], color=colors, alpha=0.8, edgecolor='black')
ax5.set_title('Model Sparsity', fontsize=12, fontweight='bold')
ax5.set_ylabel('Sparsity (%)', fontsize=10)
ax5.set_ylim([0, 100])
ax5.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 6. 精确率和召回率
ax6 = fig.add_subplot(gs[1, 2])
x = np.arange(len(algorithms))
width = 0.35
bars1 = ax6.bar(x - width/2, df['Precision (%)'], width, label='Precision', 
                color='#1f77b4', alpha=0.8, edgecolor='black')
bars2 = ax6.bar(x + width/2, df['Recall (%)'], width, label='Recall',
                color='#d62728', alpha=0.8, edgecolor='black')
ax6.set_title('Precision vs Recall', fontsize=12, fontweight='bold')
ax6.set_ylabel('Percentage (%)', fontsize=10)
ax6.set_xticks(x)
ax6.set_xticklabels(algorithms)
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# 7. 动态准确率对比
ax7 = fig.add_subplot(gs[2, :2])
for name, result in results.items():
    windowed_acc = result['windowed'].accuracy()
    windows = range(1, len(windowed_acc) + 1)
    ax7.plot(windows, windowed_acc, marker='o', label=name, 
             linewidth=2.5, markersize=5, alpha=0.8)
ax7.set_title('Accuracy Evolution over Time Windows', fontsize=12, fontweight='bold')
ax7.set_xlabel('Window Number', fontsize=10)
ax7.set_ylabel('Accuracy (%)', fontsize=10)
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3)
ax7.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Peak Features')

# 8. 综合雷达图
ax8 = fig.add_subplot(gs[2, 2], projection='polar')
categories = ['Accuracy', 'Kappa', 'F1-Score', 'Sparsity']
N = len(categories)

oasf_scores = [
    df.loc[0, 'Accuracy (%)'],
    df.loc[0, 'Kappa'] * 10,
    df.loc[0, 'F1-Score (%)'],
    df.loc[0, 'Sparsity (%)']
]

rsol_scores = [
    df.loc[1, 'Accuracy (%)'],
    df.loc[1, 'Kappa'] * 10,
    df.loc[1, 'F1-Score (%)'],
    df.loc[1, 'Sparsity (%)']
]

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
oasf_scores += oasf_scores[:1]
rsol_scores += rsol_scores[:1]
angles += angles[:1]

ax8.plot(angles, oasf_scores, 'o-', linewidth=2, label='OASF', color='#ff7f0e')
ax8.fill(angles, oasf_scores, alpha=0.25, color='#ff7f0e')
ax8.plot(angles, rsol_scores, 'o-', linewidth=2, label='RSOL', color='#2ca02c')
ax8.fill(angles, rsol_scores, alpha=0.25, color='#2ca02c')
ax8.set_xticks(angles[:-1])
ax8.set_xticklabels(categories, size=9)
ax8.set_ylim(0, 100)
ax8.set_title('Overall Performance Radar', fontsize=12, fontweight='bold', pad=20)
ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax8.grid(True)

plt.suptitle('OASF vs RSOL: Comprehensive Comparison on Feature Evolution Scenario',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('oasf_vs_rsol_comparison.png', dpi=300, bbox_inches='tight')
print("Comprehensive visualization saved to 'oasf_vs_rsol_comparison.png'")
plt.show()

print("\n" + "="*70)
print("Comparison Complete!")
print("="*70)