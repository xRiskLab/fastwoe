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

Other examples: `{"special_codes": [-999]}` (a bin of its own for a 'no record' code, kept out of the intervals), `{"binning_method": "kbins", "binner_kwargs": {"n_bins": 5}}`, `{"binning_method": "faiss_kmeans", "faiss_kwargs": {"k": 5}}`, `{"monotonic_cst": {"bureau": -1}}`.

Use few, coarse bins. Each node estimates a distribution over its feature's bins from the earlier features' bins, so every extra bin thins the cells on both sides. On a bank case study (24,859 applications, six features), validation log loss with FastWoe tree bins was:

| Bins per feature | SoftmaxWoe | Logistic regression on WOE |
|---|---|---|
| At most 4 (`{"tree_kwargs": {"max_leaf_nodes": 4}}`) | **0.1873** | 0.1885 |
| FastWoe default (up to 8) | 0.1894 | 0.1890 |
| Up to 16 (`{"tree_kwargs": {"max_depth": 4}}`) | 0.1947 | **0.1866** |

SoftmaxWoe was best with 4 bins, where it beat the scorecard; with 16 the scorecard was better. Start with about 4 bins per feature (`binning_kwargs={"tree_kwargs": {"max_leaf_nodes": 4}}`) and go finer only if cross-validation says so.

### Your Own Binner

Any object with `fit(X, y)` and `transform(X)` can supply the bins instead of FastWoe: the `Binner` protocol (`from fastwoe.softmax_woe import Binner`), which a binner satisfies by having the two methods, without inheriting from anything. `transform` must return a DataFrame or array with X's columns and a bin label in every cell: strings, numbers, intervals or categoricals.

- Ordered categoricals keep their order; other labels are sorted, numbers and intervals numerically.
- Missing values become the level `"Missing"`.
- A column the binner leaves unchanged is treated as categories (a warning flags one with more than 100 distinct values).
- The binner is cloned before fitting, so the object you pass is never modified, and `fit` may return `None`.
- It always receives every column seen at fit; where `node_proba` is given only some, the others are passed as NaN.

For example, cut points taken from a gradient-boosting model's splits (`booster.trees_to_dataframe()` in xgboost lists them per feature) or from an existing scorecard:

```python
class EdgesBinner:
    """Fixed cut points per column, as [a, b) intervals (xgboost sends x < split left)."""

    def __init__(self, edges):
        self.edges = edges

    def fit(self, X, y):
        pass

    def transform(self, X):
        out = X.copy()
        for col, cuts in self.edges.items():
            out[col] = pd.cut(X[col], [-np.inf, *cuts, np.inf], right=False)
        return out


model = SoftmaxWoe(binner=EdgesBinner({"bureau": [480, 570, 625], "utilization": [56, 78, 95]}))
```

A scikit-learn transformer works as well, for example `KBinsDiscretizer(encode="ordinal")` on numeric columns. Pass either `binner` or `binning_kwargs`, not both. With a custom binner, `binner_` is that fitted binner rather than a FastWoe.

### High-Cardinality Categories

A nominal feature with hundreds of values (postcode, occupation code, merchant) gives each value its own bin, and most of those bins are thin. Start by pooling the rare values with fastwoe's `WoePreprocessor`, which keeps frequent categories under their own names and puts the rest in one `"__other__"` bin. Set `min_count` to 30 or more: its default (10) keeps too many thin categories for SoftmaxWoe.

```python
from sklearn.pipeline import make_pipeline
from fastwoe import SoftmaxWoe, WoePreprocessor

model = make_pipeline(WoePreprocessor(min_count=30), SoftmaxWoe())
```

For more accuracy, at the cost of bins that are sets of categories rather than named ones, group the values by their smoothed risk instead. A CatBoost encoder ([`category_encoders.CatBoostEncoder`](https://contrib.scikit-learn.org/category_encoders/catboost.html)) supplies that risk: each value's event rate, shrunk toward the overall rate with strength `a`. The binner below groups those rates into a few bins and leaves the other columns to FastWoe:

```python
import numpy as np
from category_encoders import CatBoostEncoder
from fastwoe import FastWoe, SoftmaxWoe


class RiskGroupBinner:
    """FastWoe bins for most columns; listed nominal columns grouped by smoothed event rate."""

    def __init__(self, nominal, n_groups=8, a=30.0):
        self.nominal, self.n_groups, self.a = nominal, n_groups, a

    def fit(self, X, y):
        others = [c for c in X.columns if c not in self.nominal]
        self.fastwoe_ = FastWoe().fit(X[others], y)
        values = X[self.nominal].astype(str)
        self.encoder_ = CatBoostEncoder(cols=self.nominal, a=self.a).fit(values, y)
        rates = self.encoder_.transform(values)  # without y: full-sample smoothed rates
        cuts = np.linspace(0, 1, self.n_groups + 1)[1:-1]
        self.edges_ = {c: np.unique(np.quantile(rates[c], cuts)) for c in self.nominal}

    def transform(self, X):
        others = [c for c in X.columns if c not in self.nominal]
        out = self.fastwoe_.transform_bins(X[others])
        rates = self.encoder_.transform(X[self.nominal].astype(str))
        for c in self.nominal:
            out[c] = np.searchsorted(self.edges_[c], rates[c].to_numpy(), side="right")
        return out[X.columns]


model = SoftmaxWoe(binner=RiskGroupBinner(nominal=["postcode"]))
```

A value unseen at fit gets the overall rate and lands in a middle group. On simulated data with 400 postcodes whose risk depends on 8 hidden regions, validation log loss with the code above was:

| Postcode handling | Levels | Log loss |
|---|---|---|
| Every postcode its own bin | 400 | 0.407 |
| `WoePreprocessor(min_count=10)`, the default | 269 | 0.399 |
| `WoePreprocessor(min_count=30)` (frequent postcodes kept, rare ones pooled) | 63 | 0.388 |
| Grouped by smoothed risk, 8 groups, `a=30` | 7 | **0.380** |
| True model | | 0.368 |

Use it only for unordered categories. For ordered features (a bureau score, income, age) the tree bins are much better, because they pool neighboring values, where risk-grouping treats each distinct value as its own category with a thin, noisy rate. On the bank case study, risk-grouping the numeric features gave 0.208 to 0.234 against 0.187 with FastWoe's tree bins.

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

They are computed the first time they are asked for, so fitting does not pay for them. Each node's log-probability gets a delta-method variance: from the bin counts for the first node, $(1 - p) / (n p)$, and from the inverse penalized Hessian of its logistic regression after that. A weight's variance is the sum of its event and non-event nodes, which are fitted on separate rows. Within a class the log-likelihood is a sum of node terms with separate coefficients, so the nodes' estimates are asymptotically independent, and the variance of the log-odds is simply

$$\text{Var}(\text{score}) = \frac{1}{n_1} + \frac{1}{n_0} + \sum_i \text{Var}(W_i)$$

with no covariance terms. In simulation, with true weights known, 95% intervals covered them 94.5–95.0% of the time per feature at C = 100, and `predict_ci` covered the true probability 95.3% of the time.

The weights are penalized, so they are biased toward marginal WOE, and the interval covers sampling noise, not that bias. At C = 1 in the same simulation the bias was about a third of a standard deviation and coverage fell to 93%. With a strong penalty, read the interval as one under the model's assumptions.

Compared with `FastWoe(conditional=True)`: where no conditioning step falls back, its intervals come from the counts of a single cell and are wider than SoftmaxWoe's, which borrows strength across paths. Where a step falls back to the marginal weight, the interval is narrower, but the point estimate carries the double counting that conditioning was meant to remove.

## Choosing C

`C` is the inverse L2 penalty of every node. Choose it by cross-validated log loss. SoftmaxWoe is a scikit-learn estimator, so `GridSearchCV(SoftmaxWoe(order=[...]), {"C": [0.01, 0.1, 1, 10]}, scoring="neg_log_loss")` does it in one line; written out by hand:

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
- **One-hot node inputs.** Each node sees the earlier features one-hot. Feeding them in as weights of evidence instead (each earlier feature's WOE for the node's target bins) is feasible and, with a single earlier feature, gives the same fit. On the bank case study it was as accurate as one-hot with coarse bins (4 to 8 per feature) but less accurate with finer bins (validation log loss 0.1994 against 0.1947 at up to 16), where many cells are thin and their WOE estimates noisy; it was also 2 to 8 times slower to fit, because a feature with fewer bins than the node's target gives redundant, badly scaled columns. Capping WOE at ±3 halved the fitting time but lowered accuracy further, so the nodes use one-hot.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `order` | column order of X | Order in which features are conditioned on |
| `C` | 1.0 | Inverse L2 penalty of every node model |
| `root_pseudo_count` | 0.5 | Pseudo-count per bin wherever shares are counted |
| `max_iter` | 5000 | Iteration limit of each node's logistic regression |
| `unseen` | `"warn"` | `"warn"`, `"prior"` or `"raise"` for categories absent at fit |
| `binning_kwargs` | `None` | Keyword arguments for the FastWoe that bins numerical features |
| `binner` | `None` | Any object with `fit(X, y)` and `transform(X)` that returns bins; FastWoe by default |

## Fitted Attributes

| Attribute | Description |
|---|---|
| `order_` | Conditioning order used |
| `levels_` | Bins of each feature, in the order the chain uses them |
| `prior_log_odds_` | Log-odds of the event in the training data |
| `class_counts_` | Training rows per class, `{1: events, 0: non-events}` |
| `classes_` | The class labels, `[0, 1]` |
| `nodes_` | Node models keyed by `(feature, class)`: bin shares for the first node, `LogisticRegression` after |
| `binner_` | The fitted binner: a FastWoe by default, otherwise a clone of the one passed |

## References

- Good, I. J. (1950). *Probability and the Weighing of Evidence*. London: Griffin.
- Good, I. J. (1983). *Good Thinking: The Foundations of Probability and Its Applications*. Minneapolis: University of Minnesota Press.
- Goodfellow, I., Bengio, Y. and Courville, A. (2016). *Deep Learning*, §12.4.3.2, Hierarchical Softmax. MIT Press.
- Ng, A. Y. and Jordan, M. I. (2002). On discriminative vs. generative classifiers: a comparison of logistic regression and naive Bayes. *Advances in Neural Information Processing Systems* 14.
