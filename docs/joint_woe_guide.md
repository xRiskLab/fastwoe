# Joint WOE Usage Guide

## Overview

**JointWoe keeps the form of a WOE scorecard, one weight per bin of every feature, but fits all the tables together, so evidence that correlated features share is counted once.**

```text
log-odds = intercept + w_1[bin_1] + w_2[bin_2] + ... + w_p[bin_p]
```

Marginal WOE (`FastWoe`) measures each feature's table on its own. When two features carry the same signal, both tables carry it, and adding them counts it twice. JointWoe fits the tables at once, so each weight is the evidence of that bin *given the other features' tables*.

The model is logistic regression on the one-hot bins (every bin, no reference level) with an L2 penalty on the weights and none on the intercept, fitted with scikit-learn's lbfgs on a sparse matrix. At the penalized optimum every table sums to zero over its bins.

## Where it sits

| | Marginal WOE (`FastWoe`) | `JointWoe` | `SoftmaxWoe` |
|---|---|---|---|
| Each weight is | the bin's evidence on its own | the bin's evidence given the other features | the bin's evidence given the earlier features |
| Fitted by | counting | one penalized logistic regression | a chain of penalized models per class |
| Kind | descriptive | discriminative | generative |
| Correlated features | counted several times | counted once | counted once |
| Many features | fine | fine | estimation error builds up along the chain |
| Feature order | does not matter | does not matter | changes the score |

A common practice is a logistic regression on marginal WOE values: it also corrects for double counting, but with one coefficient per feature that scales the whole marginal table. JointWoe fits a weight per bin, so the shape of a table can change when another feature already explains part of it.

## Example

Two noisy readings of one underlying driver, and an independent feature:

```python
import numpy as np
import pandas as pd
from scipy.special import expit
from fastwoe import FastWoe, JointWoe

rng = np.random.default_rng(0)
n = 20_000
risk = rng.normal(size=n)
X = pd.DataFrame({
    "utilization": risk + 0.4 * rng.normal(size=n),
    "late_payments": risk + 0.4 * rng.normal(size=n),
    "income": rng.normal(size=n),
})
y = rng.binomial(1, expit(-2 + 1.5 * risk - 0.5 * X["income"]))
train, test = np.arange(n) < 15_000, np.arange(n) >= 15_000

marginal = FastWoe().fit(X[train], y[train])
joint = JointWoe(C=1.0).fit(X[train], y[train])
```

On the 5,000 test rows:

| Score | AUC | Log loss |
|---|---|---|
| Sum of marginal WOE (no regression) | 0.798 | 0.443 |
| Logistic regression on marginal WOE | 0.802 | 0.394 |
| `JointWoe` | 0.803 | 0.394 |

Summing marginal WOE double counts the shared driver and is badly calibrated. The two regressions predict equally well here; what JointWoe adds is per-bin weights that account for the other features. The conditional information value, `joint.iv(X, y)` (the mean weight over events minus the mean over non-events), shows how the shared evidence is divided:

| Feature | Marginal IV | Joint IV |
|---|---|---|
| utilization | 1.326 | 0.757 |
| late_payments | 1.294 | 0.656 |
| income | 0.128 | 0.174 |

The two readings of the same driver each lose about half their marginal IV, since each carries what the other already says. Income gains a little: with the driver accounted for, its own evidence is clearer.

`get_mapping` shows a feature's table next to its marginal WOE:

```python
joint.get_mapping("utilization")
```

```text
    category  count  event_rate  marginal_woe    woe  woe_se
  (-∞, -0.6] 4376.0       0.034        -1.936 -1.427   0.101
(-0.6, -0.1] 2681.0       0.093        -0.881 -0.721   0.073
 (-0.1, 0.4] 2342.0       0.169        -0.190 -0.275   0.060
  (0.4, 0.6] 1304.0       0.230         0.194 -0.076   0.067
  (0.6, 1.1] 2055.0       0.315         0.625  0.182   0.053
  (1.1, 1.4]  816.0       0.438         1.151  0.486   0.073
  (1.4, 1.9]  825.0       0.531         1.526  0.658   0.075
    (1.9, ∞)  601.0       0.709         2.292  1.172   0.101
```

The joint table keeps the marginal ordering but is flatter, about half as steep: the rest of the evidence is in `late_payments`.

## API

| Method | Returns |
|---|---|
| `fit(X, y, sample_weight=None)` | `y` holds 0/1 labels or proportions in [0, 1] |
| `transform(X)` | each row's weight per feature; `intercept_` plus the row sum is the log-odds |
| `transform(X, output="se")` | the standard error of each of those weights |
| `decision_function(X)` | log-odds |
| `predict_proba(X)`, `predict(X)` | as in scikit-learn |
| `predict_ci(X, alpha=0.05)` | interval for P(y = 1), columns [lower, upper] |
| `iv(X, y)` | conditional information value of each feature |
| `get_mapping(feature)` | bin, count, event rate, marginal WOE, joint weight and its SE |

`tables_` holds each feature's weights as a Series indexed by bin, `levels_` the bins, `binner_` the fitted binner. Binning works as in `SoftmaxWoe`: a `FastWoe` configured by `binning_kwargs` by default (decision-tree bins for numerical features, missing values as the bin "Missing"), or any object with `fit(X, y)` and `transform(X)` passed as `binner=`. JointWoe is a scikit-learn estimator: `clone`, `GridSearchCV` (choose `C` this way), `cross_val_score` and `Pipeline` work.

### Soft targets

`y` may be a proportion, such as another model's probabilities. JointWoe then fits the cross-entropy to it, which distills the model into a scorecard; the bins come from the default FastWoe, which accepts proportions too. A proportion per row with `sample_weight` is the same likelihood as the 0/1 rows it summarizes.

## Standard errors

The covariance of the weights is the inverse of the penalized Hessian, computed on first use.

- `predict_ci` uses the full covariance, so the covariance between features' weights is included: two correlated features' weights are uncertain individually but their sum is not.
- `transform(X, output="se")` and `get_mapping` report each weight's SE as it is reported, centered within its table. Shifting a whole table and the intercept in opposite directions changes no prediction; only the penalty decides that level, and its uncertainty is not part of a weight's meaning.

The weights are penalized, so biased toward 0; the intervals cover sampling noise, not that bias. In tests the weight SEs matched the bootstrap spread, and with one feature and a negligible penalty `predict_ci` reproduces the exact interval logit(rate) ± z √(1/events + 1/non-events).

## Correlated features

The score is well determined however correlated the features are. How the evidence is split between two near-duplicate features is not: the data cannot tell them apart, and the L2 penalty splits it evenly. Exact duplicates get identical tables, each carrying half. Read the conditional IV of strongly correlated features as a split chosen by the penalty, not a finding about either feature.
