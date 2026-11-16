# Asymptotic Standard Error of Somers' D

References: Agresti, A., 2002. *Categorical Data Analysis*, 2nd ed., sec. 3.4. Goktas, A., Oznur, I., 2011. *A Comparison of the Most Commonly Used Measures of Association for Doubly Ordered Square Contingency Tables via Simulation.* Metodoloski zvezki 8 (1), 17-37.

## 1. Setup

Given $n$ paired observations $(y_i, x_i)$, build the contingency table $\mathbf{F}$ where rows correspond to sorted unique values of $Y$ and columns to sorted unique values of $X$.

| Symbol | Definition |
|--------|-----------|
| $f_{ij}$ | Cell count: number of observations in row $i$, column $j$ |
| $a$ | Number of rows (unique $Y$ values) |
| $b$ | Number of columns (unique $X$ values) |
| $r_i$ | Row sum: $r_i = \sum_j f_{ij}$ |
| $W$ | Sample size: $W = \sum_{ij} f_{ij} = n$ |

## 2. Concordant and Discordant Counts

For each cell $(i, j)$ of the contingency table, define:

**$C_{ij}$** = number of observations that would form a **concordant** pair with any observation in cell $(i, j)$. These are observations in cells strictly above-left or strictly below-right:

$$C_{ij} = \sum_{\substack{r > i \\ s > j}} f_{rs} \;+\; \sum_{\substack{r < i \\ s < j}} f_{rs}$$

**$D_{ij}$** = number of observations that would form a **discordant** pair:

$$D_{ij} = \sum_{\substack{r > i \\ s < j}} f_{rs} \;+\; \sum_{\substack{r < i \\ s > j}} f_{rs}$$

These are computed efficiently using 2D prefix sums.

## 3. Somers' D Statistic

Total concordant and discordant pairs:

$$P = \sum_{ij} f_{ij} \cdot C_{ij} \qquad Q = \sum_{ij} f_{ij} \cdot D_{ij}$$

Denominator - number of ordered pairs untied on $Y$:

$$D_r = W^2 - \sum_i r_i^2$$

For binary $Y$ with $m$ positives and $W - m$ negatives: $D_r = 2m(W - m)$.

**Somers' D:**

$$D = \frac{P - Q}{D_r}$$

For binary targets, this equals the Gini coefficient $= 2 \cdot \text{AUC} - 1$.

## 4. ASE via the Delta Method

Somers' D is a **ratio of two U-statistics**: $D = (P - Q) / D_r$.

To get the variance of a ratio $f/g$, the delta method gives:

$$\text{Var}\!\left(\frac{f}{g}\right) \approx \frac{1}{g^2} \cdot \text{Var}(f - D \cdot g)$$

For each cell $(i, j)$, the **linearized residual** has two parts:

### Part 1: Numerator contribution (scaled)

$$D_r \cdot (C_{ij} - D_{ij})$$

This is how much cell $(i, j)$ contributes to the net concordance, scaled by the denominator.

### Part 2: Denominator correction

$$(P - Q) \cdot (W - r_i)$$

One more observation in row $i$ adds $2(W - r_i)$ to $D_r$: it is untied on $Y$ with every observation outside its row. The factor $(P - Q) = D \cdot D_r$ weights this by the current value of $D$.

> **Correction (0.3.1).** Earlier versions used the row midrank $R_i = \sum_{k \le i} r_k + (1 - r_i)/2$ in place of the row count $r_i$, following the VUROCS R package. With the midrank the term is not the derivative of $D_r$: the standard error did not shrink with the sample size (about 0.08 at $n = 5{,}000$, where the bootstrap gives 0.027), and reversing the order of $Y$ changed it. With $r_i$ it agrees with DeLong's standard error and with the bootstrap, and is unchanged when $Y$ is reversed.

### Combined residual per cell

$$\varepsilon_{ij} = D_r \cdot (C_{ij} - D_{ij}) \;-\; (P - Q) \cdot (W - r_i)$$

The second term depends only on the row, not the column: the denominator $D_r$ is determined entirely by the $Y$ marginal distribution.

### ASE formula

The weighted sum of squared residuals, normalized:

$$\boxed{\text{ASE}_1 = \frac{2}{D_r^2} \sqrt{\sum_{ij} f_{ij} \cdot \varepsilon_{ij}^2}}$$

Expanding:

$$\text{ASE}_1 = \frac{2}{D_r^2} \sqrt{\sum_{ij} f_{ij} \left\{ D_r(C_{ij} - D_{ij}) - (P - Q)(W - r_i) \right\}^2}$$

The factor of 2 accounts for the symmetry of ordered pairs (each unordered pair is counted twice).

## 5. ASE Under the Null ($\text{ASE}_0$)

Under $H_0\!: D = 0$, the denominator correction term vanishes (since $P - Q = 0$), giving a simpler formula:

$$\boxed{\text{ASE}_0 = \frac{2}{D_r} \sqrt{\sum_{ij} f_{ij} (C_{ij} - D_{ij})^2 \;-\; \frac{(P - Q)^2}{W}}}$$

This is used for hypothesis testing (is $D$ significantly different from zero?).

## 6. Summary

| Quantity | Formula | Use |
|----------|---------|-----|
| $D$ | $(P - Q) \;/\; D_r$ | Point estimate |
| $\text{ASE}_1$ | $\frac{2}{D_r^2} \sqrt{\sum f_{ij} \cdot \varepsilon_{ij}^2}$ | Confidence intervals |
| $\text{ASE}_0$ | $\frac{2}{D_r} \sqrt{\sum f_{ij}(C_{ij}-D_{ij})^2 - (P-Q)^2/W}$ | Hypothesis testing |

For a 95% confidence interval:

$$D \;\pm\; z_{0.975} \cdot \text{ASE}_1$$

For a two-sided test of $H_0\!: D = 0$:

$$z = \frac{D}{\text{ASE}_0}$$

## 7. Worked Example

A small binary credit risk table: 10 applicants, target $Y \in \{0, 1\}$, WOE-encoded feature $X \in \{1, 2, 3\}$ (three bins).

```python
import numpy as np

#-----------------------------------------------------------------------------------------------
# Data
#-----------------------------------------------------------------------------------------------
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
x = np.array([1, 1, 2, 2, 3, 1, 2, 3, 3, 3], dtype=float)

#-----------------------------------------------------------------------------------------------
# Step 1: Contingency table
#-----------------------------------------------------------------------------------------------
#          x=1  x=2  x=3
#   y=0  [  2    2    1 ]
#   y=1  [  1    1    3 ]
CT = np.array([[2, 2, 1],
               [1, 1, 3]], dtype=float)

a, b = CT.shape          # 2 rows, 3 columns
W = CT.sum()              # 10
r = CT.sum(axis=1)        # [5, 5]

print(f"CT:\n{CT}")
print(f"W={W}, r={r}")

#-----------------------------------------------------------------------------------------------
# Step 2: Concordant (C) and discordant (D) matrices
#-----------------------------------------------------------------------------------------------
# C[i,j] = sum of cells strictly below-right + strictly above-left
# D[i,j] = sum of cells strictly below-left  + strictly above-right
C = np.zeros_like(CT)
D = np.zeros_like(CT)
for i in range(a):
    for j in range(b):
        C[i, j] = CT[i+1:, j+1:].sum() + CT[:i, :j].sum()
        D[i, j] = CT[i+1:, :j].sum()  + CT[:i, j+1:].sum()

print(f"\nC (concordant counts):\n{C}")
print(f"D (discordant counts):\n{D}")

#-----------------------------------------------------------------------------------------------
# Step 3: Somers' D
#-----------------------------------------------------------------------------------------------
P  = (CT * C).sum()                # total concordant
Q  = (CT * D).sum()                # total discordant
Dr = W**2 - (r**2).sum()           # pairs untied on Y

D_val = (P - Q) / Dr

print(f"\nP={P}, Q={Q}, Dr={Dr}")
print(f"Somers' D = (P-Q)/Dr = {D_val:.6f}")

#-----------------------------------------------------------------------------------------------
# Step 4: ASE (delta method)
#-----------------------------------------------------------------------------------------------
# Linearized residual per cell; W - r[i] = observations outside row i
eps = Dr * (C - D) - (P - Q) * (W - r)[:, np.newaxis]
print(f"\nResiduals eps:\n{eps}")

ASE = 2.0 / Dr**2 * np.sqrt((CT * eps**2).sum())
print(f"\nASE = {ASE:.6f}")

#-----------------------------------------------------------------------------------------------
# Step 5: ASE under H0
#-----------------------------------------------------------------------------------------------
ASE0 = 2.0 / Dr * np.sqrt((CT * (C - D)**2).sum() - (P - Q)**2 / W)
print(f"ASE0 = {ASE0:.6f}")

#-----------------------------------------------------------------------------------------------
# Step 6: 95% confidence interval
#-----------------------------------------------------------------------------------------------
print(f"\n95% CI: [{D_val - 1.96*ASE:.4f}, {D_val + 1.96*ASE:.4f}]")
print(f"z-test (H0: D=0): z = {D_val / ASE0:.4f}")
```

Output:

```
CT:
[[2. 2. 1.]
 [1. 1. 3.]]
W=10.0, r=[5. 5.]

C (concordant counts):
[[4. 3. 0.]
 [0. 2. 4.]]
D (discordant counts):
[[0. 1. 2.]
 [3. 1. 0.]]

P=28.0, Q=8.0, Dr=50.0

Somers' D = (P-Q)/Dr = 0.400000

Residuals eps:
[[ 100.    0. -200.]
 [-250.  -50.  100.]]

ASE = 0.314960
ASE0 = 0.314960

95% CI: [-0.2173, 1.0173]
z-test (H0: D=0): z = 1.2700
```

## 8. Production Implementation

See `fastwoe.metrics.somersd_se()`. The tests check it against this worked example, against DeLong's standard error and the bootstrap for binary targets, and that reversing $Y$ leaves it unchanged. (ASE and ASE0 coincide in the example above because both rows have the same count; they differ in general.)
