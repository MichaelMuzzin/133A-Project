<h3 style="color:#7216e6">(a) Basic Linear Model</h3>

**Operations:**

- Basis Functions: No expansion is done. The model uses the original standardized features.
- Prediction: For each sample, the model computes a dot product between the feature vector (length d) and the coefficient vector and then adds an intercept.
 • Dot product: roughly d multiplications + d – 1 additions (`≈ 2d flops per sample`).
 • Plus a constant cost for the exponential inverse transformation.

Total per sample: `~2d flops` (ignoring constant overhead).

Overall: `~O(n·d)`.

Observed RMS error: `≈ 160,217`

<h3 style="color:#7216e6">(b) Feature Engineering with K-Means Clustering</h3>

**Operations:**

1. Cluster Assignment:
    - For each sample, the model computes the `squared Euclidean distance to each of 3 centroids`.
    - Each distance requires about 2d flops (d multiplications and roughly d additions); for 3 clusters, that’s roughly `6d` flops.

2. Prediction:
    - Now the feature vector has length `(d + 3)`, so the dot product costs about `2·(d + 3)` flops.

Total per sample: `~6d` (clustering) `+ 2*(d + 3)` (dot product) `≈ 8d + constant`.

Overall: `~O(n·d)` with a higher constant factor than (a).

Observed RMS error: `≈ 145,051`

<h3 style="color:#7216e6">(c) Regularized Ridge Model with Expanded Basis Functions</h3>

**Operations:**

2nd-Order Polynomial Expansion:
1. The PolynomialFeatures (degree 2, no bias) transforms the original d features into:
    - d original (linear) terms, plus 
    - `d(d+1)1/2` quadratic terms.
    Thus, the new number of features is
    - `p = d + d(d+1)/2`
    **The cost to compute the quadratic terms is roughly on the order of d(d+1)/2 multiplications per sample (we assume the linear terms are “copied” with negligible cost).**
2. Appending Cluster Dummies:
    - Adds 3 extra features (constant cost per sample).
3. Prediction:
    - The model computes a dot product with a feature vector of length `p + 3`, costing about `2·(p + 3)` flops per sample.

Total per sample:
- Basis expansion: `~d(d+1)/2 flops, i.e. O(d²)`.
- Dot product: `~2·(d + d(d+1)/2 + 3) ≈ O(d²)`.

Overall: `~O(n·d²)`.

Observed RMS error: `≈ 137,528`
Note: Although the per-sample cost is higher than in (a) and (b), the **model achieves the lowest RMS error** among these options.

<h3 style="color:#7216e6">(d) Non-Linear Model with Degree-3 Polynomial Expansion</h3>

**Operations:**

1. 3rd-Order Polynomial Expansion:
    The expansion converts `d` features into a set that includes:
    - d linear terms,
    - d(d+1)/2 quadratic terms, and
    - d(d+1)(d+2)/6 cubic terms.

    Thus, the total number of features is
`q = d + d(d+1)/2 + d(d+1)(d+2)/6`
    The cost for this basis function computation is on the order of `O(d³)` flops per sample.

2. Prediction:
    The dot product now is computed over a feature vector of length roughly `q + 3`, costing about `2·(q + 3)` flops per sample (again, `O(d³)`).

Overall: `~O(n·d³)`.

Observed RMS error: `≈ 125,755,957`
**Note:** Despite the dramatic increase in computational cost, the RMS error is far worse. This suggests that the excessive complexity (and likely numerical instability or overfitting) of a `degree-3 expansion` does not yield a better model.







