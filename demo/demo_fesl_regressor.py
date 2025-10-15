"""Demo script for FESL Regressor - Simplified Version"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

from capymoa.datasets import Fried
from capymoa.evaluation import prequential_evaluation
from capymoa.regressor import FESLRegressor

# Load stream
fried_stream = Fried()
schema = fried_stream.schema

print(f"Number of features: {schema.get_num_attributes()}")

# Create FESL regressor with feature spaces
# S1: features 0-5, S2: features 4-9 (overlap on 4-5)
fesl_learner = FESLRegressor(
    schema=schema,
    s1_feature_indices=list(range(0, 6)),
    s2_feature_indices=list(range(4, 10)),
    overlap_size=100,
    switch_point=5000,
    ensemble_method="combination",
    learning_rate_scale=0.1,
    random_seed=42
)

# Run evaluation with progress bar
results = prequential_evaluation(
    stream=fried_stream,
    learner=fesl_learner,
    max_instances=10000,
    window_size=100,
    progress_bar=True
)

# Print final results
print(f"Ensemble Method: {fesl_learner.ensemble_method}")
print(f"\nFinal MAE: {results['cumulative'].mae():.4f}")
print(f"Final RMSE: {results['cumulative'].rmse():.4f}")