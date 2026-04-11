# Asymptotic Standard Error of Somers' D

Reference: Goktas, A., Oznur, I., 2011. *A Comparison of the Most Commonly Used Measures of Association for Doubly Ordered Square Contingency Tables via Simulation.* Metodoloski zvezki 8 (1), 17-37.

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

## 4. Row Midranks

Define the midrank of row $i$ as the average position of observations in that row:

$$R_i = \sum_{k=1}^{i} r_k + \frac{1 - r_i}{2}$$

This places each row at the center of its range in the marginal ranking.

For example, if $r = [3, 3, 3]$:

- $R_1 = 3 + (1-3)/2 = 2$
- $R_2 = 6 + (1-3)/2 = 5$
- $R_3 = 9 + (1-3)/2 = 8$

## 5. ASE via the Delta Method

Somers' D is a **ratio of two U-statistics**: $D = (P - Q) / D_r$.

To get the variance of a ratio $f/g$, the delta method gives:

$$\text{Var}\!\left(\frac{f}{g}\right) \approx \frac{1}{g^2} \cdot \text{Var}(f - D \cdot g)$$

For each cell $(i, j)$, the **linearized residual** has two parts:

### Part 1: Numerator contribution (scaled)

$$D_r \cdot (C_{ij} - D_{ij})$$

This is how much cell $(i, j)$ contributes to the net concordance, scaled by the denominator.

### Part 2: Denominator correction

$$(P - Q) \cdot (W - R_i)$$

Here $(W - R_i)$ captures how much row $i$ contributes to $D_r$ (pairs untied on $Y$). The factor $(P - Q) = D \cdot D_r$ weights this by the current value of $D$.

### Combined residual per cell

$$\varepsilon_{ij} = D_r \cdot (C_{ij} - D_{ij}) \;-\; (P - Q) \cdot (W - R_i)$$

Note that $R_i$ depends only on the row, not the column - the denominator $D_r$ is determined entirely by the $Y$ marginal distribution.

### ASE formula

The weighted sum of squared residuals, normalized:

$$\boxed{\text{ASE}_1 = \frac{2}{D_r^2} \sqrt{\sum_{ij} f_{ij} \cdot \varepsilon_{ij}^2}}$$

Expanding:

$$\text{ASE}_1 = \frac{2}{D_r^2} \sqrt{\sum_{ij} f_{ij} \left\{ D_r(C_{ij} - D_{ij}) - (P - Q)(W - R_i) \right\}^2}$$

The factor of 2 accounts for the symmetry of ordered pairs (each unordered pair is counted twice).

## 6. ASE Under the Null ($\text{ASE}_0$)

Under $H_0\!: D = 0$, the denominator correction term vanishes (since $P - Q = 0$), giving a simpler formula:

$$\boxed{\text{ASE}_0 = \frac{2}{D_r} \sqrt{\sum_{ij} f_{ij} (C_{ij} - D_{ij})^2 \;-\; \frac{(P - Q)^2}{W}}}$$

This is used for hypothesis testing (is $D$ significantly different from zero?).

## 7. Summary

| Quantity | Formula | Use |
|----------|---------|-----|
| $D$ | $(P - Q) \;/\; D_r$ | Point estimate |
| $\text{ASE}_1$ | $\frac{2}{D_r^2} \sqrt{\sum f_{ij} \cdot \varepsilon_{ij}^2}$ | Confidence intervals |
| $\text{ASE}_0$ | $\frac{2}{D_r} \sqrt{\sum f_{ij}(C_{ij}-D_{ij})^2 - (P-Q)^2/W}$ | Hypothesis testing |

For a 95% confidence interval:

$$D \;\pm\; z_{0.975} \cdot \text{ASE}_1$$

For a two-sided test of $H_0\!: D = 0$:

$$z = \frac{D}{\text{ASE}_0}$$

## 8. Worked Example

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
# Step 4: Row midranks
#-----------------------------------------------------------------------------------------------
RR = np.cumsum(r) + (1.0 - r) / 2.0
print(f"\nRow midranks R = {RR}")

#-----------------------------------------------------------------------------------------------
# Step 5: ASE (delta method)
#-----------------------------------------------------------------------------------------------
RR_mat = np.repeat(RR[:, np.newaxis], b, axis=1)

# Linearized residual per cell
eps = Dr * (C - D) - (P - Q) * (W - RR_mat)
print(f"\nResiduals eps:\n{eps}")

ASE = 2.0 / Dr**2 * np.sqrt((CT * eps**2).sum())
print(f"\nASE = {ASE:.6f}")

#-----------------------------------------------------------------------------------------------
# Step 6: ASE under H0
#-----------------------------------------------------------------------------------------------
ASE0 = 2.0 / Dr * np.sqrt((CT * (C - D)**2).sum() - (P - Q)**2 / W)
print(f"ASE0 = {ASE0:.6f}")

#-----------------------------------------------------------------------------------------------
# Step 7: 95% confidence interval
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

Row midranks R = [3. 8.]

Residuals eps:
[[  60.  -40. -240.]
 [-190.   10.  160.]]

ASE = 0.340353
ASE0 = 0.314960

95% CI: [-0.2671, 1.0671]
z-test (H0: D=0): z = 1.2700
```

## 9. Production Implementation

See `fastwoe.metrics.somersd_se()` - validated against the VUROCS R package with exact numerical agreement across binary, ordinal, and continuous targets.
