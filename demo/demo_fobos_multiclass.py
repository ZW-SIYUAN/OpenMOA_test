"""demo_fobos_multiclass.py - FOBOS Multi-class Classifier Demo
Demonstrates group sparsity and mixed-norm regularization.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Covtype
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import FOBOSMulticlassClassifier
import numpy as np


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def analyze_weight_matrix(W, feature_names=None):
    """Analyze and visualize weight matrix structure."""
    n_features, n_classes = W.shape
    
    # Row sparsity analysis
    row_norms = np.linalg.norm(W, axis=1)
    zero_rows = np.sum(row_norms < 1e-8)
    nonzero_rows = n_features - zero_rows
    
    print(f"\n📊 Weight Matrix Structure:")
    print(f"  Shape:                 {n_features} features × {n_classes} classes")
    print(f"  Total parameters:      {W.size}")
    print(f"  Frobenius norm:        {np.linalg.norm(W, 'fro'):.4f}")
    
    print(f"\n🎯 Row Sparsity (Feature Selection):")
    print(f"  Zero rows:             {zero_rows}/{n_features} ({zero_rows/n_features:.1%})")
    print(f"  Non-zero rows:         {nonzero_rows}/{n_features} ({nonzero_rows/n_features:.1%})")
    print(f"  → {nonzero_rows} features are active across all classes")
    
    # Element sparsity
    zero_elements = np.sum(np.abs(W) < 1e-8)
    print(f"\n📈 Element-wise Sparsity:")
    print(f"  Zero elements:         {zero_elements}/{W.size} ({zero_elements/W.size:.1%})")
    print(f"  Non-zero elements:     {W.size - zero_elements}/{W.size}")
    
    # Row norm distribution
    if nonzero_rows > 0:
        nonzero_row_norms = row_norms[row_norms > 1e-8]
        print(f"\n📉 Active Row Norms Distribution:")
        print(f"  Mean:                  {np.mean(nonzero_row_norms):.4f}")
        print(f"  Max:                   {np.max(nonzero_row_norms):.4f}")
        print(f"  Min (non-zero):        {np.min(nonzero_row_norms):.4f}")
        print(f"  Std:                   {np.std(nonzero_row_norms):.4f}")
    
    # Top important features (by row norm)
    if nonzero_rows > 0 and nonzero_rows <= 20:
        print(f"\n⭐ Top Active Features (by ||row||₂):")
        sorted_indices = np.argsort(row_norms)[::-1]
        for i, idx in enumerate(sorted_indices[:min(10, nonzero_rows)]):
            if row_norms[idx] > 1e-8:
                feature_name = f"Feature {idx}" if feature_names is None else feature_names[idx]
                print(f"  {i+1:2d}. {feature_name:<25} norm={row_norms[idx]:.4f}")


def evaluate_multiclass_learner(learner, stream, max_instances=5000, description=""):
    """Evaluate a multi-class FOBOS learner and print results."""
    print(f"\n{description}")
    print("-" * 70)
    print(f"Configuration: {learner}")
    
    results = prequential_evaluation(
        stream=stream,
        learner=learner,
        max_instances=max_instances,
        window_size=500
    )
    
    # Performance metrics
    print(f"\n📊 Performance Metrics:")
    print(f"  Accuracy:              {results['cumulative'].accuracy():.2f}%")
    print(f"  F1 Score:              {results['cumulative'].f1_score():.2f}%")
    print(f"  Precision:             {results['cumulative'].precision():.2f}%")
    print(f"  Recall:                {results['cumulative'].recall():.2f}%")
    print(f"  Kappa:                 {results['cumulative'].kappa():.4f}")
    print(f"  Instances processed:   {results['cumulative'].instances_seen}")
    
    # Analyze weight matrix
    W = learner.get_weights()
    analyze_weight_matrix(W)
    
    return results


def demonstrate_group_sparsity():
    """Demonstrate how mixed-norm regularization achieves group sparsity."""
    print_section("CONCEPT: GROUP SPARSITY (Feature-level Selection)")
    
    print("\n🎓 What is Group Sparsity?")
    print("  In multi-class learning, we have a weight matrix W ∈ R^{d×k}")
    print("  • Each row corresponds to ONE feature across ALL classes")
    print("  • Regular L1: Makes individual weights zero (w_{i,j} = 0)")
    print("  • L1/L2 mixed-norm: Makes entire ROWS zero (all w_{i,*} = 0)")
    
    print("\n📚 Example:")
    print("  Task: Classify news into 3 categories (Sports, Finance, Tech)")
    print("  Features: 'basketball', 'stock', 'algorithm', 'weather', ...")
    
    print("\n  With L1/L2 regularization:")
    print("    'basketball' row → [0.8, 0.1, 0.0]  (active)")
    print("    'stock' row      → [0.0, 0.9, 0.1]  (active)")
    print("    'algorithm' row  → [0.0, 0.2, 0.9]  (active)")
    print("    'weather' row    → [0.0, 0.0, 0.0]  (ENTIRE ROW IS ZERO!)")
    
    print("\n  → Result: 'weather' is irrelevant for ALL classes")
    print("  → This is FEATURE SELECTION at the feature level, not weight level")
    
    print("\n💡 Why is this useful?")
    print("  ✓ Interpretability: Know which features matter")
    print("  ✓ Efficiency: Ignore zero-row features at test time")
    print("  ✓ Generalization: Reduce overfitting by removing useless features")


def main():
    """Run comprehensive FOBOS multi-class classification demo."""
    
    print_section("FOBOS MULTI-CLASS CLASSIFIER - COMPREHENSIVE DEMO")
    print("\nThis demo showcases:")
    print("  • Multi-class classification with softmax loss")
    print("  • Mixed-norm regularization (L1/L2, L1/L∞)")
    print("  • Group sparsity for feature selection")
    print("\nDataset: CoverType (581,012 instances, 54 features, 7 classes)")
    
    # Explain group sparsity concept
    demonstrate_group_sparsity()
    
    # ========================================================================
    # EXPERIMENT 1: L1/L2 Mixed-norm (Recommended)
    # ========================================================================
    print_section("EXPERIMENT 1: L1/L2 MIXED-NORM (RECOMMENDED)")
    
    stream = Covtype()
    learner_l1_l2 = FOBOSMulticlassClassifier(
        schema=stream.get_schema(),
        alpha=1.0,
        lambda_=0.1,  # Strong regularization to see sparsity
        regularization="l1_l2",
        step_schedule="sqrt",
        random_seed=42
    )
    
    results_l1_l2 = evaluate_multiclass_learner(
        learner_l1_l2,
        stream,
        max_instances=10000,
        description="🔸 L1/L2 Mixed-norm - Group sparsity for feature selection"
    )
    
    # ========================================================================
    # EXPERIMENT 2: L1/L∞ Mixed-norm
    # ========================================================================
    print_section("EXPERIMENT 2: L1/L∞ MIXED-NORM")
    
    stream = Covtype()
    learner_l1_linf = FOBOSMulticlassClassifier(
        schema=stream.get_schema(),
        alpha=1.0,
        lambda_=0.1,
        regularization="l1_linf",
        step_schedule="sqrt",
        random_seed=42
    )
    
    results_l1_linf = evaluate_multiclass_learner(
        learner_l1_linf,
        stream,
        max_instances=10000,
        description="🔸 L1/L∞ Mixed-norm - Alternative group sparsity method"
    )
    
    # ========================================================================
    # EXPERIMENT 3: Lower Regularization (Less Sparse)
    # ========================================================================
    print_section("EXPERIMENT 3: LOWER REGULARIZATION (LESS SPARSE)")
    
    stream = Covtype()
    learner_low_reg = FOBOSMulticlassClassifier(
        schema=stream.get_schema(),
        alpha=1.0,
        lambda_=0.01,  # Much lower regularization
        regularization="l1_l2",
        step_schedule="sqrt",
        random_seed=42
    )
    
    results_low_reg = evaluate_multiclass_learner(
        learner_low_reg,
        stream,
        max_instances=10000,
        description="🔸 Lower λ - More features retained, less sparsity"
    )
    
    # ========================================================================
    # EXPERIMENT 4: Linear Step Schedule
    # ========================================================================
    print_section("EXPERIMENT 4: LINEAR STEP SCHEDULE")
    
    stream = Covtype()
    learner_linear = FOBOSMulticlassClassifier(
        schema=stream.get_schema(),
        alpha=1.0,  # Higher alpha for linear schedule
        lambda_=0.1,
        regularization="l1_l2",
        step_schedule="linear",
        random_seed=42
    )
    
    results_linear = evaluate_multiclass_learner(
        learner_linear,
        stream,
        max_instances=10000,
        description="🔸 Linear schedule (η_t=α/t) - Faster convergence"
    )
    
    # ========================================================================
    # COMPARATIVE SUMMARY
    # ========================================================================
    print_section("COMPARATIVE SUMMARY")
    
    experiments = [
        ("L1/L2 (λ=0.1)", results_l1_l2, learner_l1_l2),
        ("L1/L∞ (λ=0.1)", results_l1_linf, learner_l1_linf),
        ("L1/L2 (λ=0.01)", results_low_reg, learner_low_reg),
        ("Linear schedule", results_linear, learner_linear)
    ]
    
    print("\n📊 Performance Comparison:")
    print(f"{'Method':<20} {'Accuracy':<12} {'F1':<12} {'Row Sparsity':<15} {'Elem Sparsity'}")
    print("-" * 80)
    
    for name, result, learner in experiments:
        acc = result['cumulative'].accuracy()
        f1 = result['cumulative'].f1_score()
        row_sparsity = learner.get_row_sparsity()
        elem_sparsity = learner.get_element_sparsity()
        print(f"{name:<20} {acc:>10.2f}%  {f1:>10.2f}%  {row_sparsity:>13.1%}  {elem_sparsity:>13.1%}")
    
    # ========================================================================
    # KEY INSIGHTS
    # ========================================================================
    print_section("KEY INSIGHTS")
    
    print("\n💡 Mixed-norm Regularization:")
    print("  • L1/L2: Most common, good balance of sparsity and stability")
    print("  • L1/L∞: Alternative, may be better for specific problems")
    print("  • Higher λ → More row sparsity (fewer active features)")
    print("  • Lower λ → More features retained (better accuracy, less sparse)")
    
    print("\n💡 Group Sparsity Benefits:")
    print("  ✓ Feature Selection: Identify important features across ALL classes")
    print("  ✓ Interpretability: Understand which features drive predictions")
    print("  ✓ Efficiency: Compute only non-zero rows at test time")
    print("  ✓ Generalization: Reduce overfitting with fewer features")
    
    print("\n💡 Comparison to Binary Classification:")
    print("  • Binary: Element-wise sparsity (individual weights → 0)")
    print("  • Multi-class: Row sparsity (entire features → 0)")
    print("  • Multi-class needs MORE data (k times more parameters)")
    print("  • Multi-class training is SLOWER (k times computation)")
    
    # ========================================================================
    # WEIGHT MATRIX VISUALIZATION
    # ========================================================================
    print_section("WEIGHT MATRIX VISUALIZATION (L1/L2 Example)")
    
    W = learner_l1_l2.get_weights()
    n_features, n_classes = W.shape
    
    print(f"\n📊 Weight Matrix Heatmap (Top 10 features):")
    print(f"  Rows = Features, Columns = Classes")
    print(f"  Values shown: weight magnitudes\n")
    
    # Get top 10 features by row norm
    row_norms = np.linalg.norm(W, axis=1)
    top_indices = np.argsort(row_norms)[::-1][:10]
    
    # Header
    print(f"  {'Feature':<15}", end="")
    for c in range(n_classes):
        print(f" Class{c}", end="")
    print(f"  ||row||₂")
    print("  " + "-" * (15 + 8*n_classes + 10))
    
    # Rows
    for idx in top_indices:
        print(f"  Feature {idx:<7}", end="")
        for c in range(n_classes):
            val = W[idx, c]
            if abs(val) < 1e-8:
                print(f"    ·   ", end="")
            else:
                print(f" {val:>6.3f}", end="")
        print(f"  {row_norms[idx]:.4f}")
    
    print("\n  Legend: '·' = zero weight (< 1e-8)")
    
    # ========================================================================
    # DETAILED MODEL INFORMATION
    # ========================================================================
    print_section("DETAILED MODEL INFORMATION (L1/L2 Example)")
    print("\n" + learner_l1_l2.get_model_description())
    
    # ========================================================================
    # PRACTICAL RECOMMENDATIONS
    # ========================================================================
    print_section("PRACTICAL RECOMMENDATIONS")
    
    print("\n🎯 When to use FOBOS Multi-class:")
    print("  ✓ Need feature selection across all classes")
    print("  ✓ High-dimensional multi-class problems")
    print("  ✓ Want interpretable models (which features matter?)")
    print("  ✓ Online/streaming multi-class classification")
    
    print("\n⚠️  When NOT to use:")
    print("  ✗ Small number of features (< 50) - overhead not worth it")
    print("  ✗ Need maximum accuracy - use ensemble methods instead")
    print("  ✗ Extreme class imbalance - may need specialized handling")
    
    print("\n🔧 Hyperparameter Tuning Guide:")
    print("  • λ (regularization):")
    print("    - Start: λ = 0.01")
    print("    - More sparsity: Increase to 0.05-0.1")
    print("    - Less sparsity: Decrease to 0.001-0.005")
    print("  • α (learning rate):")
    print("    - sqrt schedule: α = 1.0-5.0")
    print("    - linear schedule: α = 5.0-10.0")
    print("  • regularization:")
    print("    - Default: 'l1_l2' (most common)")
    print("    - Try 'l1_linf' if L1/L2 doesn't work well")
    
    print_section("DEMO COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()