"""demo_fobos_binary.py - FOBOS Binary Classifier Demo
Demonstrates all regularization types and learning configurations.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Electricity
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import FOBOSClassifier
import numpy as np


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def evaluate_learner(learner, stream, max_instances=5000, description=""):
    """Evaluate a FOBOS learner and print results."""
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
    
    # Sparsity analysis
    sparsity = learner.get_sparsity()
    weights = learner.get_weights()
    nonzero_count = np.sum(np.abs(weights) > 1e-8)
    
    print(f"\n🎯 Sparsity Analysis:")
    print(f"  Sparsity level:        {sparsity:.2%}")
    print(f"  Zero weights:          {len(weights) - nonzero_count}/{len(weights)}")
    print(f"  Non-zero weights:      {nonzero_count}/{len(weights)}")
    
    # Weight statistics
    if nonzero_count > 0:
        nonzero_weights = weights[np.abs(weights) > 1e-8]
        print(f"\n📈 Weight Statistics:")
        print(f"  Mean |w|:              {np.mean(np.abs(weights)):.4f}")
        print(f"  Max |w|:               {np.max(np.abs(weights)):.4f}")
        print(f"  Min |w| (non-zero):    {np.min(np.abs(nonzero_weights)):.4f}")
        print(f"  L2 norm:               {np.linalg.norm(weights):.4f}")
    
    return results


def main():
    """Run comprehensive FOBOS binary classification demo."""
    
    print_section("FOBOS BINARY CLASSIFIER - COMPREHENSIVE DEMO")
    print("\nThis demo showcases all features of FOBOSClassifier:")
    print("  • Multiple regularization types (L1, L2, L2², Elastic Net)")
    print("  • Different loss functions (Logistic, Hinge)")
    print("  • Various learning rate schedules")
    print("\nDataset: Electricity (45,312 instances, 8 features, binary)")
    
    # ========================================================================
    # EXPERIMENT 1: L1 Regularization (Sparse Learning)
    # ========================================================================
    print_section("EXPERIMENT 1: L1 REGULARIZATION (SPARSE LEARNING)")
    
    stream = Electricity()
    learner_l1 = FOBOSClassifier(
        schema=stream.get_schema(),
        alpha=1.0,
        lambda_=0.05,  # Strong regularization for sparsity
        regularization="l1",
        step_schedule="sqrt",
        loss="logistic",
        random_seed=42
    )
    
    results_l1 = evaluate_learner(
        learner_l1, 
        stream, 
        max_instances=10000,
        description="🔸 L1 Regularization - Promotes sparse solutions (many zero weights)"
    )
    
    # ========================================================================
    # EXPERIMENT 2: L2² Regularization (Ridge Regression Style)
    # ========================================================================
    print_section("EXPERIMENT 2: L2² REGULARIZATION (RIDGE STYLE)")
    
    stream = Electricity()
    learner_l2sq = FOBOSClassifier(
        schema=stream.get_schema(),
        alpha=1.0,
        lambda_=0.05,
        regularization="l2_squared",
        step_schedule="sqrt",
        loss="logistic",
        random_seed=42
    )
    
    results_l2sq = evaluate_learner(
        learner_l2sq,
        stream,
        max_instances=10000,
        description="🔸 L2² Regularization - Smooth shrinkage, no sparsity"
    )
    
    # ========================================================================
    # EXPERIMENT 3: Elastic Net (L1 + L2² Combination)
    # ========================================================================
    print_section("EXPERIMENT 3: ELASTIC NET (L1 + L2²)")
    
    stream = Electricity()
    learner_en = FOBOSClassifier(
        schema=stream.get_schema(),
        alpha=1.0,
        lambda_=0.05,
        regularization="elastic_net",
        elastic_net_ratio=0.7,  # 70% L1, 30% L2²
        step_schedule="sqrt",
        loss="logistic",
        random_seed=42
    )
    
    results_en = evaluate_learner(
        learner_en,
        stream,
        max_instances=10000,
        description="🔸 Elastic Net (70% L1, 30% L2²) - Balanced sparse + stable"
    )
    
    # ========================================================================
    # EXPERIMENT 4: L2 Regularization (Spherical Shrinkage)
    # ========================================================================
    print_section("EXPERIMENT 4: L2 REGULARIZATION (SPHERICAL SHRINKAGE)")
    
    stream = Electricity()
    learner_l2 = FOBOSClassifier(
        schema=stream.get_schema(),
        alpha=1.0,
        lambda_=0.05,
        regularization="l2",
        step_schedule="sqrt",
        loss="logistic",
        random_seed=42
    )
    
    results_l2 = evaluate_learner(
        learner_l2,
        stream,
        max_instances=10000,
        description="🔸 L2 Regularization - Entire weight vector shrinkage"
    )
    
    # ========================================================================
    # EXPERIMENT 5: Hinge Loss (SVM-style)
    # ========================================================================
    print_section("EXPERIMENT 5: HINGE LOSS (SVM-STYLE)")
    
    stream = Electricity()
    learner_svm = FOBOSClassifier(
        schema=stream.get_schema(),
        alpha=0.5,  # Lower learning rate for hinge loss
        lambda_=0.01,
        regularization="l2_squared",
        step_schedule="sqrt",
        loss="hinge",
        random_seed=42
    )
    
    results_svm = evaluate_learner(
        learner_svm,
        stream,
        max_instances=10000,
        description="🔸 Hinge Loss + L2² - SVM-style online learning"
    )
    
    # ========================================================================
    # EXPERIMENT 6: Linear Step Schedule (Strongly Convex)
    # ========================================================================
    print_section("EXPERIMENT 6: LINEAR STEP SCHEDULE")
    
    stream = Electricity()
    learner_linear = FOBOSClassifier(
        schema=stream.get_schema(),
        alpha=5.0,  # Larger alpha for linear schedule
        lambda_=0.05,
        regularization="l2_squared",
        step_schedule="linear",  # η_t = α/t
        loss="logistic",
        random_seed=42
    )
    
    results_linear = evaluate_learner(
        learner_linear,
        stream,
        max_instances=10000,
        description="🔸 Linear schedule (η_t=α/t) - For strongly convex problems"
    )
    
    # ========================================================================
    # COMPARATIVE SUMMARY
    # ========================================================================
    print_section("COMPARATIVE SUMMARY")
    
    experiments = [
        ("L1", results_l1, learner_l1),
        ("L2²", results_l2sq, learner_l2sq),
        ("Elastic Net", results_en, learner_en),
        ("L2", results_l2, learner_l2),
        ("Hinge+L2²", results_svm, learner_svm),
        ("Linear Schedule", results_linear, learner_linear)
    ]
    
    print("\n📊 Performance Comparison:")
    print(f"{'Method':<20} {'Accuracy':<12} {'F1':<12} {'Sparsity':<12} {'||w||₂'}")
    print("-" * 70)
    
    for name, result, learner in experiments:
        acc = result['cumulative'].accuracy()
        f1 = result['cumulative'].f1_score()
        sparsity = learner.get_sparsity()
        norm = np.linalg.norm(learner.get_weights())
        print(f"{name:<20} {acc:>10.2f}%  {f1:>10.2f}%  {sparsity:>10.1%}  {norm:>8.4f}")
    
    # ========================================================================
    # KEY INSIGHTS
    # ========================================================================
    print_section("KEY INSIGHTS")
    
    print("\n💡 Regularization Trade-offs:")
    print("  • L1: Highest sparsity, good for feature selection")
    print("  • L2²: No sparsity, stable convergence, good for correlated features")
    print("  • Elastic Net: Balance between sparsity and stability")
    print("  • L2: Entire vector shrinkage, rare sparsity")
    
    print("\n💡 Loss Function Characteristics:")
    print("  • Logistic: Probabilistic predictions, smooth gradient")
    print("  • Hinge: SVM-style, non-probabilistic, sparse support")
    
    print("\n💡 Learning Rate Schedule:")
    print("  • sqrt (η_t=α/√t): General convex, O(√T) regret")
    print("  • linear (η_t=α/t): Strongly convex, O(log T) regret")
    
    # ========================================================================
    # DETAILED MODEL INFORMATION
    # ========================================================================
    print_section("DETAILED MODEL INFORMATION (L1 Example)")
    print("\n" + learner_l1.get_model_description())
    
    print_section("DEMO COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()