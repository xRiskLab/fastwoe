# Softmax WOE Usage Guide

## Overview

**SoftmaxWoe is a generative classifier: per class, an autoregressive chain of penalized multinomial logistic regressions, fitted by maximum likelihood of the features given the class; its per-feature weights are Good's conditional weights of evidence under that model and sum exactly to the posterior log-odds.**

It is the smoothed counterpart of `FastWoe(conditional=True)`. Both give each feature the evidence it adds *given the features before it*, so correlated features are not counted twice. Conditional WOE reads that evidence off cell counts, which run thin after a few features. SoftmaxWoe fits a small model per feature instead, so every applicant gets a conditional weight however rare their profile.

## Background

### Conditional weights of evidence

I. J. Good's chain rule splits the evidence of several features into a sum of conditional weights:

$$W(H : E_1 E_2 \ldots E_k) = W(H : E_1) + W(H : E_2 \mid E_1) + \ldots + W(H : E_k \mid E_1 \ldots E_{k-1})$$

Each weight is a log ratio of likelihoods, measured among applicants who share the earlier evidence:

$$W(H : E_i \mid E_{<i}) = \log P(E_i \mid E_{<i}, H) - \log P(E_i \mid E_{<i}, \bar H)$$

Marginal WOE drops the conditioning and adds $W(H : E_i)$ for every feature, which double counts whatever signal the features share. Conditional WOE keeps it, by counting each bin within the cell of earlier bins. With six features of three or four bins there are hundreds of cells, and many hold only a handful of the rarer class.

### One model per feature

SoftmaxWoe estimates each $P(E_i \mid E_{<i}, H)$ with a multinomial logistic regression (a *node*): its target is the bin of feature $i$, its inputs are the bins of the earlier features, one-hot coded. Each node is fitted twice, once among events and once among non-events. The first node has nothing to condition on, so it is the bin shares with a 0.5 pseudo-count.

Per class this is the hierarchical softmax of Goodfellow et al. (2016, §12.4.3.2), with applicant profiles in place of words: one level per feature, and the probability of a profile is the product of the node probabilities along its path. Every node is a proper distribution over its bins, so each class's chain is a proper distribution over all profiles, and the weights telescope:

$$\text{log-odds} = \log \frac{P(H)}{P(\bar H)} + \sum_i W(H : E_i \mid E_{<i})$$

Nothing is left over and nothing needs recalibrating.

### A ladder of assumptions

The methods differ only in what each node may see:

| Node for feature *i*, within each class | Assumes | Result |
|---|---|---|
| Bin shares only | Feature *i* is unrelated to the earlier bins | Marginal WOE (naive Bayes) |
| Logistic regression on earlier bins | Earlier bins act additively on the log scale | **SoftmaxWoe** |
| Every combination of earlier bins | Nothing | Conditional WOE (counting) |

The penalty `C` moves along the ladder: as `C` goes to 0 every node shrinks to its bin shares and SoftmaxWoe becomes marginal WOE; as `C` grows the nodes fit the main effects of the earlier bins freely.

### Generative, not discriminative

SoftmaxWoe maximizes the likelihood of the features given the class, $\sum \log P(x \mid y)$, and gets the posterior by Bayes' rule. Logistic regression on WOE maximizes $\sum \log P(y \mid x)$ directly. This is the classic generative/discriminative pair (Ng & Jordan, 2002, compared naive Bayes with logistic regression). In the examples below and on real credit data the two reach the same log loss, from opposite directions: logistic regression by shrinking each marginal weight after the first, SoftmaxWoe by conditioning it.

Only the fully saturated counting chain is also the conditional maximum-likelihood fit, because with every cell free the two criteria give the same answer. That is why a logistic regression fitted on top of saturated conditional WOE learns slopes of 1.

## Basic Usage

### 1. Fit and Transform

```python
import numpy as np
import pandas as pd
from fastwoe import SoftmaxWoe

# Three correlated features: utilization and the cheque card both follow the bureau score
rng = np.random.default_rng(0)
n = 20_000
y = rng.binomial(1, 0.1, n)
bureau = rng.normal(640 - 60 * y, 60)
utilization = np.clip(rng.normal(70 - 0.08 * (bureau - 600) + 10 * y, 15), 0, 100)
card = np.where(rng.random(n) < 1 / (1 + np.exp(-(bureau - 600) / 40)), "Y", "N")
X = pd.DataFrame({"bureau": bureau, "utilization": utilization, "card": card})
X.loc[rng.random(n) < 0.03, "bureau"] = np.nan

train = rng.random(n) < 0.7
X_train, y_train, X_test, y_test = X[train], y[train], X[~train], y[~train]

model = SoftmaxWoe(order=["bureau", "utilization", "card"], C=1.0)
model.fit(X_train, y_train)

W = model.transform(X_test)  # one conditional weight per feature
print(W.head().round(3))
```

Output:

```
    bureau  utilization   card
3   -1.364       -0.125 -0.094
5    3.885        1.080  0.412
12  -0.045        0.002 -0.128
13  -0.045       -0.885  0.082
22  -1.364       -1.242 -0.016
```

### 2. Score

The prior plus the row sum is the log-odds of the event, and `predict_proba` is its sigmoid:

```python
log_odds = model.prior_log_odds_ + W.sum(axis=1)   # prior_log_odds_ = -2.202
proba = model.predict_proba(X_test)[:, 1]
```

```
log-odds:  -3.785   3.176  -2.372  -3.050  -4.824
P(event):  0.0222  0.9599  0.0853  0.0452  0.0080
```

### 3. Compare with Marginal WOE

```python
from fastwoe import FastWoe
from sklearn.linear_model import LogisticRegression

fw = FastWoe().fit(X_train, y_train)
lr = LogisticRegression(penalty=None).fit(fw.transform(X_train), y_train)
```

| Model | Test log loss |
|---|---|
| Marginal WOE (sum of weights) | 0.2904 |
| Logistic regression on marginal WOE | 0.2682 |
| SoftmaxWoe | 0.2684 |

Mean absolute weight per feature:

| Feature | Marginal WOE | SoftmaxWoe |
|---|---|---|
| bureau | 0.806 | 0.806 |
| utilization | 0.858 | 0.663 |
| card | 0.495 | 0.101 |

The first feature keeps its marginal weight, as it must. The card is mostly a restatement of the bureau score, so once the score is known it adds a fifth of its marginal weight. Marginal WOE counts that shared signal again and is overconfident; logistic regression repairs the score by shrinking coefficients, SoftmaxWoe by conditioning the weights, which still add up to the score.

## Bins

### Numerical Features

Numerical features are binned exactly as FastWoe bins them (decision-tree bins by default, for numeric columns with at least 20 distinct values). Other columns, and numeric columns with fewer distinct values, are treated as categories. Missing values form their own level, `"Missing"`. The bins of each feature are in `levels_`, in numeric order:

```python
model.levels_["bureau"]
```

```
['(-∞, 429.5]', '(429.5, 478.6]', '(478.6, 545.5]', '(545.5, 572.3]',
 '(572.3, 595.6]', '(595.6, 625.2]', '(625.2, 658.2]', '(658.2, ∞)', 'Missing']
```

The fitted FastWoe is kept as `binner_`: `binner_.get_binning_summary()` shows the bins, `binner_.transform()` gives the marginal WOE, and `binner_.transform_bins(X)` gives each value's bin.

### Configuring the Binning

`binning_kwargs` passes keyword arguments to that FastWoe, so every FastWoe binning option is available:

```python
model = SoftmaxWoe(
    order=["bureau", "utilization", "card"],
    binning_kwargs={"tree_kwargs": {"max_leaf_nodes": 4}},
)
model.fit(X_train, y_train)
model.levels_["bureau"]
# ['(-∞, 478.6]', '(478.6, 572.3]', '(572.3, 625.2]', '(625.2, ∞)', 'Missing']
```

Other examples: `{"binning_method": "kbins", "binner_kwargs": {"n_bins": 5}}`, `{"binning_method": "faiss_kmeans", "faiss_kwargs": {"k": 5}}`, `{"monotonic_cst": {"bureau": -1}}`.

Each node has a coefficient per earlier bin, so fewer, coarser bins mean steadier nodes. Capping the tree is a good default: on a bank case study (24,859 applications, six features) four tree bins per feature gave a lower validation log loss than both the default tree bins and hand-picked cut points.

## Inspecting a Node

`node_proba` returns, for each row, the probability of its bin at that feature's node in each class. The log of their ratio is the feature's weight:

```python
p = model.node_proba(X_test.head(3), "utilization")
p.assign(weight=np.log(p.p_event / p.p_nonevent))
```

```
    p_event  p_nonevent  weight
3    0.3220      0.3649  -0.125
5    0.3833      0.1301   1.080
12   0.1712      0.1708   0.002
```

`X` needs the feature and every feature before it in `order_`. The node models themselves are scikit-learn `LogisticRegression` objects in `nodes_`, keyed by `(feature, class)`; their inputs are the earlier features one-hot in `levels_` order.

## Standard Errors

`transform(X, output="se")` gives the standard error of every weight, and `predict_ci(X, alpha=0.05)` an interval for the probability, as `[lower, upper]` like `FastWoe.predict_ci`:

```python
model.transform(X_test, output="se").head()
model.predict_ci(X_test)[:5]
```

```
    bureau  utilization   card        P(event)   lower    upper
3    0.084        0.126  0.046          0.0222   0.0163   0.0302
5    0.688        0.948  0.409          0.9599   0.6775   0.9963
12   0.064        0.152  0.102          0.0853   0.0598   0.1205
13   0.064        0.229  0.109          0.0452   0.0275   0.0734
22   0.084        0.246  0.036          0.0080   0.0048   0.0133
```

Each node's log-probability gets a delta-method variance: from the bin counts for the first node, $(1 - p) / (n p)$, and from the inverse penalized Hessian of its logistic regression after that. A weight's variance is the sum of its event and non-event nodes, which are fitted on separate rows. Within a class the log-likelihood is a sum of node terms with separate coefficients, so the nodes' estimates are asymptotically independent, and the variance of the log-odds is simply

$$\text{Var}(\text{score}) = \frac{1}{n_1} + \frac{1}{n_0} + \sum_i \text{Var}(W_i)$$

with no covariance terms. In simulation, with true weights known, 95% intervals covered them 94.5–95.0% of the time per feature at C = 100, and `predict_ci` covered the true probability 95.3% of the time.

The weights are penalized, so they are biased toward marginal WOE, and the interval covers sampling noise, not that bias. At C = 1 in the same simulation the bias was about a third of a standard deviation and coverage fell to 93%. With a strong penalty, read the interval as one under the model's assumptions.

Compared with `FastWoe(conditional=True)`: where no conditioning step falls back, its intervals come from the counts of a single cell and are wider than SoftmaxWoe's, which borrows strength across paths. Where a step falls back to the marginal weight, the interval is narrower, but the point estimate carries the double counting that conditioning was meant to remove.

## Choosing C

`C` is the inverse L2 penalty of every node. Choose it by cross-validated log loss:

```python
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

def cv_log_loss(C, X, y, n_splits=5):
    scores = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=0).split(X, y):
        m = SoftmaxWoe(order=["bureau", "utilization", "card"], C=C).fit(X.iloc[tr], y[tr])
        scores[te] = m.predict_proba(X.iloc[te])[:, 1]
    return log_loss(y, scores)

for C in (0.001, 0.01, 0.1, 1.0, 10.0):
    print(C, round(cv_log_loss(C, X_train, y_train), 4))
```

```
0.001   0.2767
0.01    0.2713
0.1     0.2697
1.0     0.2700
10.0    0.2706
```

The curve is usually flat over a wide range of `C`, as here, and only clearly worse when `C` is small enough to push the nodes back toward marginal WOE.

The penalty is applied to one coefficient row per bin. Where a node has only two bins, scikit-learn fits a single logit instead, and SoftmaxWoe fits it at `2C` so that two-bin nodes are shrunk exactly like the rest.

## Order

`order` fixes which features each weight is conditioned on. The first feature gets its marginal weight; each later one gets what it adds given the earlier ones. Reversing the order moves the evidence between features:

| Feature | Mean \|weight\|, forward | Mean \|weight\|, reversed |
|---|---|---|
| bureau | 0.806 | 0.529 |
| utilization | 0.663 | 0.792 |
| card | 0.101 | 0.494 |

Unlike counted conditional WOE, where every order lands in the same cell and only the attribution changes, the order also changes SoftmaxWoe's score, because each order defines a different smoothed distribution. Here the probabilities differ by 0.0007 on average, but by up to 0.18 for the most affected applicant. Choose the order by reasoning (bureau data before application data, cause before symptom) and report it with the reason codes.

## Notes

- **Binary targets only.** The target must be 0/1 with both classes present.
- **Unseen categories** at transform get weight 0 and a warning (`unseen="warn"`); `unseen="prior"` does the same silently and `unseen="raise"` fails. Numerical values always fall into a bin.
- **One-class bins.** In a bin that holds only one class, marginal WOE is extreme (FastWoe's smoothing is tiny). The first node's 0.5 pseudo-count (`root_pseudo_count`) tempers it.
- **Monotonic constraints** shape the bins, so they hold for marginal WOE and for the first feature in `order`. The node models are not constrained, so a later feature's conditional weights need not be monotone. Put a constrained feature first if that matters.
- **No temperature, no recalibration.** Dividing the node logits or the final score by a temperature did not improve on T = 1 in our experiments: the fitted chain is already calibrated, and `C` is the right shrinkage knob.
- **Penalty type.** L1 and elastic-net nodes reached the same log loss as L2 at their cross-validated `C`; L2 is used because it degrades more gently when `C` is mis-set.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `order` | column order of X | Order in which features are conditioned on |
| `C` | 1.0 | Inverse L2 penalty of every node model |
| `root_pseudo_count` | 0.5 | Pseudo-count per bin wherever shares are counted |
| `max_iter` | 5000 | Iteration limit of each node's logistic regression |
| `unseen` | `"warn"` | `"warn"`, `"prior"` or `"raise"` for categories absent at fit |
| `binning_kwargs` | `None` | Keyword arguments for the FastWoe that bins numerical features |

## Fitted Attributes

| Attribute | Description |
|---|---|
| `order_` | Conditioning order used |
| `levels_` | Bins of each feature, in the order the chain uses them |
| `prior_log_odds_` | Log-odds of the event in the training data |
| `class_counts_` | Training rows per class, `{1: events, 0: non-events}` |
| `nodes_` | Node models keyed by `(feature, class)`: bin shares for the first node, `LogisticRegression` after |
| `binner_` | The fitted FastWoe that supplies the bins |

## References

- Good, I. J. (1950). *Probability and the Weighing of Evidence*. London: Griffin.
- Good, I. J. (1983). *Good Thinking: The Foundations of Probability and Its Applications*. Minneapolis: University of Minnesota Press.
- Goodfellow, I., Bengio, Y. and Courville, A. (2016). *Deep Learning*, §12.4.3.2, Hierarchical Softmax. MIT Press.
- Ng, A. Y. and Jordan, M. I. (2002). On discriminative vs. generative classifiers: a comparison of logistic regression and naive Bayes. *Advances in Neural Information Processing Systems* 14.
