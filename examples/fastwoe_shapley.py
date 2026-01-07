"""
## Shapley Value Decomposition for Gini/Somers' D Contributions

Shapley values provide a fair attribution of Gini/Somers' D contributions using **exact Shapley value computation** (enumerating all subsets). When a base score is specified, the function:

- Fixes the population to where the base score is available
- Always includes the base score in the averaged score
- Computes Shapley values exactly by enumerating all possible subsets (2^n)
- Shows base-only Gini/Somers' D separately and incremental effects for extras

The Shapley values are computed exactly (not approximated) by evaluating all subsets, ensuring:

- Order-invariant attribution
- Values sum to the total combined Gini/Somers' D
- All interactions are accounted for

Note: For binary classification, Gini coefficient equals Somers' D (2 × AUC - 1).
"""

import numpy as np

from fastwoe.screening import somersd_shapley

# You have 3 credit scoring models
np.random.seed(42)
n = 10000

# Simulate target (default = 1, no default = 0)
y = np.random.binomial(1, 0.15, n)

# Simulate scores from different models (higher score = higher risk)
bureau_score = np.random.randn(n) + 0.5 * y  # Traditional bureau score
alt_data_score = np.random.randn(n) + 0.3 * y  # Alternative data (phone, utility)
behavioral_score = np.random.randn(n) + 0.4 * y  # Behavioral patterns

scores = {"bureau": bureau_score, "alt_data": alt_data_score, "behavioral": behavioral_score}

# Compute Shapley attribution
result = somersd_shapley(scores, y)
print(result)
