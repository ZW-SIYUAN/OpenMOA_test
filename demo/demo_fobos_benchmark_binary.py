"""
demo_final_paper_benchmark.py
-----------------------------
OpenMOA Final Benchmark Runner (Corrected)
------------------------------------------
Features:
1. Fixes 'Time Scale Mismatch' by injecting correct 'total_instances' to Wrappers.
2. Runs 5 Repeats (Fixed Data Order, Varying Random Seeds).
3. Reports Mean ± Std for both Prequential (Last Window) and Cumulative Accuracy.
4. Outputs a CSV ready for LaTeX table creation.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from collections import deque

# Fix Tkinter/Matplotlib GUI errors
import matplotlib
matplotlib.use('Agg')

# Ensure src is in path
sys.path.insert(0, os.path.abspath('./src'))

try:
    from capymoa.classifier._fobos_classifier import FOBOSClassifier
    from capymoa.stream.stream_wrapper import OpenFeatureStream
    import capymoa.datasets
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# === Configuration ===
N_REPEATS = 5            # Number of runs per dataset (Seeds 1 to N)
WINDOW_SIZE = 1000       # Window size for Prequential Accuracy calculation
OUTPUT_DIR = "./experiments_final"

def get_datasets():
    """Select datasets for the paper."""
    # You can uncomment more datasets as needed
    target_list = [
        ("Australian", "Australian"), 
        ("Ionosphere", "Ionosphere"),
        ("German", "German"),
        ("SVMGuide3", "SVMGuide3"),
        ("Spambase", "Spambase"), 
        ("Magic04", "Magic04"),
        ("Musk", "Musk"),
        ("InternetAds", "InternetAds"),
        ("Adult", "Adult"),
        ("w8a", "W8a"),
        ("RCV1", "RCV1") 
    ]
    available = []
    for d_name, c_name in target_list:
        if hasattr(capymoa.datasets, c_name):
            available.append((d_name, getattr(capymoa.datasets, c_name)))
    return available

def get_stream_length(base_stream, default=10000):
    """Safely determine the total length of a stream."""
    if hasattr(base_stream, "get_num_instances"):
        return base_stream.get_num_instances()
    if hasattr(base_stream, "n_instances"):
        return base_stream.n_instances
    # Fallback for streams where length is unknown
    return default

def run_single_seed(dataset_cls, wrapper_mode, seed):
    """
    Runs a single experiment with a specific seed.
    Returns: (Final Prequential Accuracy, Final Cumulative Accuracy)
    """
    
    # 1. Initialize Base Stream
    base_stream = dataset_cls()
    n_total = get_stream_length(base_stream)
    
    # 2. Configure Wrapper with CORRECT total_instances
    # This ensures evolution (TDS/EDS) is synchronized with the stream length
    if wrapper_mode == "TDS":
        stream = OpenFeatureStream(base_stream, evolution_pattern="tds", tds_mode="ordered", 
                                   d_min=2, total_instances=n_total, random_seed=seed)
    elif wrapper_mode == "CDS":
        stream = OpenFeatureStream(base_stream, evolution_pattern="cds", missing_ratio=0.4, 
                                   total_instances=n_total, random_seed=seed)
    elif wrapper_mode == "EDS":
        stream = OpenFeatureStream(base_stream, evolution_pattern="eds", n_segments=3, 
                                   overlap_ratio=0.2, total_instances=n_total, random_seed=seed)
    else:
        raise ValueError(f"Unknown mode: {wrapper_mode}")

    schema = base_stream.get_schema()
    
    # 3. Initialize Learner
    learner = FOBOSClassifier(
        schema=schema,
        alpha=0.1,        # Robust learning rate for evolving streams
        lambda_=0.0001,   # Mild regularization
        regularization="l1",
        random_seed=seed  # Seed affects weight initialization
    )
    
    # 4. Evaluation Loop
    # Metrics
    total_seen = 0
    total_correct = 0
    window = deque(maxlen=WINDOW_SIZE)
    window_correct = 0
    
    while stream.has_more_instances():
        instance = stream.next_instance()
        if instance is None: break
        
        # Predict
        pred = learner.predict(instance)
        is_correct = (pred == instance.y_index)
        
        # Update Cumulative
        total_seen += 1
        if is_correct: total_correct += 1
        
        # Update Prequential Window
        if len(window) >= WINDOW_SIZE:
            left_out = window.popleft()
            if left_out: window_correct -= 1
        window.append(is_correct)
        if is_correct: window_correct += 1
        
        # Train
        learner.train(instance)
        
    # Calculate Final Metrics
    final_cum_acc = total_correct / total_seen if total_seen > 0 else 0.0
    final_preq_acc = window_correct / len(window) if len(window) > 0 else 0.0
    
    return final_preq_acc, final_cum_acc

def main():
    datasets = get_datasets()
    if not datasets:
        print("❌ No datasets found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    modes = ["TDS", "CDS", "EDS"]
    results = []

    print(f"🚀 Starting Final Benchmark ({N_REPEATS} Runs/Exp)")
    print(f"📂 Output will be saved to: {OUTPUT_DIR}/final_results.csv")
    print("-" * 80)

    for mode in modes:
        print(f"\n🔹 Evaluation Mode: {mode}")
        print(f"{'Dataset':<15} | {'Progress':<15} | {'Preq. Acc (Mean ± Std)':<25} | {'Cum. Acc (Mean ± Std)':<25}")
        print("-" * 90)
        
        for d_name, d_cls in datasets:
            preq_scores = []
            cum_scores = []
            
            print(f"{d_name:<15} | ", end="", flush=True)
            
            start_time = time.time()
            for seed in range(1, N_REPEATS + 1):
                p_acc, c_acc = run_single_seed(d_cls, mode, seed)
                preq_scores.append(p_acc)
                cum_scores.append(c_acc)
                print(".", end="", flush=True)
            
            # Statistics
            p_mean = np.mean(preq_scores)
            p_std = np.std(preq_scores)
            c_mean = np.mean(cum_scores)
            c_std = np.std(cum_scores)
            
            elapsed = time.time() - start_time
            
            # Display Row
            print(f" Done ({elapsed:.1f}s) | {p_mean:.2%} ± {p_std:.2%}        | {c_mean:.2%} ± {c_std:.2%}")
            
            results.append({
                "Mode": mode,
                "Dataset": d_name,
                "Preq_Mean": p_mean,
                "Preq_Std": p_std,
                "Cum_Mean": c_mean,
                "Cum_Std": c_std
            })

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "final_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Benchmark Completed. Data saved to {csv_path}")

if __name__ == "__main__":
    main()