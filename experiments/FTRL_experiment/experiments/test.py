import sys, os
sys.path.insert(0, os.path.abspath('./src'))

from pathlib import Path
from capymoa.stream import BagOfWordsStream
from capymoa.classifier import FTRLClassifier

DATA_DIR = Path("C:/reposity clone/OpenMOA/experiments/FTRL_experiment/data")

datasets = ['books', 'dvd', 'electronics', 'kitchen']
ALPHA_GRID = [0.3, 0.5, 0.8, 1.0, 1.5, 1.9]

print("="*70)
print("验证 λ = 0.05 在所有数据集上的表现")
print("="*70)

for dataset in datasets:
    pos_file = DATA_DIR / f'processed_acl/{dataset}/positive.review'
    neg_file = DATA_DIR / f'processed_acl/{dataset}/negative.review'
    
    if not (pos_file.exists() and neg_file.exists()):
        print(f"\n⚠️  {dataset}: 文件不存在")
        continue
    
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset.upper()}")
    print(f"{'='*70}")
    
    # Grid search alpha
    print(f"\n{'Alpha':<10} {'Accuracy':<12} {'Sparsity':<12}")
    print("-"*70)
    
    best_alpha = None
    best_acc = -1
    
    for alpha in ALPHA_GRID:
        stream = BagOfWordsStream(
            positive_file=pos_file,
            negative_file=neg_file,
            dataset_name=dataset,
            normalize=True,
            shuffle_seed=42
        )
        
        learner = FTRLClassifier(
            schema=stream.get_schema(),
            alpha=alpha,
            beta=1.0,
            l1=0.05,  # ✅ 固定用 0.05
            l2=1.0
        )
        
        correct = 0
        for instance in stream:
            if learner.predict(instance) == instance.y_index:
                correct += 1
            learner.train(instance)
        
        acc = correct / len(stream) * 100
        sparsity = learner.get_sparsity()
        
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
        
        print(f"{alpha:<10.1f} {acc:<12.3f} {sparsity:<12.3f}")
    
    print(f"\n✅ Best α = {best_alpha}, Accuracy = {best_acc:.3f}%")