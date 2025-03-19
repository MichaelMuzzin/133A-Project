


<h3 style="color:#7216e6">Goal of this project:</h3>
<p>The goal of this project is to utilize the topics we learned in ECE 133A to a real-world scenario of algorithms, analyzing data, computing. This includes k-means clustering and singular value decomposition amongst other topics to predict the income of an individual based on current data regarding major, income, company, etc.
Using a 2024 Stackoverflow survey with more that 70,000 responses, we able to compile a comprehensive and accurate set of data inputs. In order to make the computations easier. We narrowed the data to just United States participants and removed some questions that we deemed unnecessary to our analysis. As with the project guidlines, there are 4 individual parts to this project that we have outlined: PART 1-4

<h3 style="color:#7216e6">Students in this group:</h3>

-  Shahab Besharatlou 
-  Michael Muzzin 
-  Sina Ghadimi 


<h3 style="color:#7216e6">PART 1:</h3>



<h4 style="color:#FF5733">Most Popular Programming Languages in the U.S. (2024)</h4>


Chart Type: Horizontal Bar Chart

This chart displays the 20 most commonly used programming languages among U.S. developers in the dataset.

How It Was Generated:

* The “LanguageHaveWorkedWith” column was extracted.  
* Entries were split by semicolons to count occurrences of each language.  
* The top 20 languages were plotted in a horizontal bar chart with labels inside the bars.

Key Insights:

* The most popular languages include Python, JavaScript, SQL, and Java, indicating their widespread use in the industry.  
* The chart helps identify which programming languages are most valuable in the job market.

<h4 style="color:#FF5733">Salary Data by Age Groups</h4>


Chart Type: Bar Chart

A series of bar charts were created to compare salary distributions across different age groups.

How It Was Generated:

* Developers were grouped into age categories:  
  * Under 18  
  * 18-24 years old  
  * 25-34 years old  
  * 35-44 years old  
  * 55-64 years old  
  * Salary statistics for each group were calculated, including:  
  * Maximum salary  
  * Minimum salary  
  * Median salary  
  * Mean salary  
  * Standard deviation  
  * Each category was plotted with a separate bar chart.

Key Insights:

* Income increases with age up to a certain point, with developers aged 25-44 earning the highest salaries.  
* Young developers (18-24 years old) earn significantly lower salaries compared to those over 30\.  
* A decline in salaries after age 55 suggests that fewer respondents in this age group report high compensation.

<h4 style="color:#FF5733">3 Developer Roles & Popularity</h4>


Chart Type: Horizontal Bar Chart

This chart shows the most common job titles in the dataset.

How It Was Generated:

* The “DevType” column was used to count the number of respondents in each job category.  
* Job categories with higher representation were plotted in a horizontal bar chart.

Key Insights:

* The most common roles in the dataset include Software Developers, Web Developers, and Data Scientists.  
* Some specialized roles (e.g., Embedded Systems Engineers, DevOps Specialists) have fewer respondents.  
* Understanding role distribution helps tailor salary predictions for different job categories.

<h4 style="color:#FF5733">Salary Comparison by Career Path</h4>


Chart Type: Bar Chart

A separate bar chart was generated for each career group, showing how salaries vary based on career paths.

How It Was Generated:

* Developers were grouped based on career type.  
* Within each career group, salary metrics were calculated (max, min, median, mean, standard deviation).  
* These statistics were plotted as a bar chart.

Key Insights:

* Certain careers (e.g., Machine Learning, Cloud Computing) have significantly higher mean salaries than general software development roles.  
* Developers in low-paying fields (e.g., Technical Support, QA Testing) tend to have lower max salaries than other groups.  
* Median salaries offer a better representation than mean salaries due to the presence of outliers in some careers.

<h4 style="color:#FF5733">Salary Trends by Experience Level</h4>


Chart Type: Bar Chart

This chart visualizes salary trends based on years of experience.

How It Was Generated:

* Developers were categorized into experience levels (e.g., Entry-Level, Mid-Level, Senior).  
* Salary statistics for each level were aggregated and plotted in a bar chart.

Key Insights:

* Salary grows exponentially with experience, with senior developers earning significantly higher wages.  
* Entry-level positions have a wide salary range, indicating that experience isn’t the only factor affecting pay.  
* Some high-paying roles (e.g., AI Engineering, Cloud Architecture) require substantial experience.

<h4 style="color:#FF5733">Salary vs. Most Used Programming Languages</h4>


Chart Type: Scatter Plot

This chart explores the relationship between programming language usage and salary.

How It Was Generated:

* Developers using certain languages were grouped.  
* The average salary of developers using each language was calculated.  
* A scatter plot was generated to show how salaries vary by language.

Key Insights:

* C++, Rust, and Go developers tend to have higher average salaries than Python or JavaScript users.  
* Web development languages (e.g., JavaScript, PHP) are associated with lower salaries, likely due to lower barriers to entry.  
* Specialized languages (e.g., Swift, Kotlin, R) show more variation in salary distribution.

<h4 style="color:#FF5733">Income Distribution in the U.S.</h4>


Chart Type: Histogram

A histogram was created to visualize the distribution of salaries among U.S. developers.

How It Was Generated:

* The “CompTotal” column was used to analyze salary distribution.  
* Salaries above `$1,000,000` were excluded to remove outliers.  
* A histogram was plotted to show the frequency of salary ranges.

Key Insights:

* Most developers earn between `$50,000-$150,000`.  
* There is a long tail of high salaries, but few earn above `$500,000`.  
* The histogram suggests a log-normal distribution, meaning salary transformations (log-scale) are beneficial for predictive modeling.

<h4 style="color:#FF5733">Final Summary of Part 1:</h4>


1. Programming Languages: Python and JavaScript are the most common, but C++, Rust, and Go developers tend to earn more.  
2. Age & Salary: Salaries increase with age up to 45, then decline slightly.  
3. Job Titles: Software Developers and Data Scientists are the most common roles.  
4. Career Paths: AI, Cloud, and Machine Learning roles pay significantly higher.  
5. Experience & Salary: More experience \= higher pay, but some senior roles vary widely.  
6. Language vs. Salary: Backend and specialized languages tend to have higher salaries.  
7. Income Distribution: Most salaries fall in `$50,000-$150,000`, but some outliers exist.






<h3 style="color:#7216e6">PART 2:</h3>
<h4 style="color:#FF5733">1. Overview of Part 2</h4>


In Part 2, we focused on understanding the structure of our dataset. This involved:

1. Standardizing raw features – ensuring numerical consistency.  
2. Applying K-Means clustering – identifying natural groupings in our data.  
3. Performing Singular Value Decomposition (SVD) – finding dominant features.  
4. Analyzing feature correlations – identifying redundant or related features.

Each step was accompanied by statistical analysis and visualizations.

<h4 style="color:#FF5733">2. Data Standardization</h4>


Before performing clustering or dimensionality reduction, we needed to standardize our dataset.

Code: Standardizing Features

```python
from sklearn.preprocessing import StandardScaler

# Select only numeric features
America_numeric = America.select_dtypes(include=["number"]).dropna(axis=1, how="all")

# Standardize the dataset
scaler = StandardScaler()
America_standardized = pd.DataFrame(scaler.fit_transform(America_numeric), columns=America_numeric.columns)

# Compute mean and standard deviation
feature_stats = pd.DataFrame({
    "Mean": America_standardized.mean(),
    "Standard Deviation": America_standardized.std()
})
print("\nFeature Statistics (Standardized Data):\n", feature_stats)
```

Key Insights:

* Standardization ensures that all features have a mean of 0 and a standard deviation of 1\.  
* This prevents features with larger magnitudes from dominating models like K-Means or SVD.

<h4 style="color:#FF5733">3. Handling Missing Data</h4>


Missing values can distort clustering and SVD results, so we imputed them using the mean of each feature.

Code: Imputing Missing Values

```python
from sklearn.impute import SimpleImputer

# Handle missing values by imputing the mean
imputer = SimpleImputer(strategy="mean")
America_standardized_imputed = pd.DataFrame(imputer.fit_transform(America_standardized),
                                            columns=America_standardized.columns)
```

Key Insights:

* This approach replaces NaN values with the column mean, ensuring that no data is lost.  
* Mean imputation is a safe default but might not be ideal for skewed distributions.

<h4 style="color:#FF5733">4. K-Means Clustering (Finding Natural Groups)</h4>


Purpose:

* Group respondents into clusters based on their standardized numerical features.  
* Determine the optimal number of clusters (k) using the Elbow Method.

Code: Finding Optimal k

```python
from sklearn.cluster import KMeans

# Perform K-Means Clustering with different k values
inertia = []
k_values = range(2, 11)  # Testing k from 2 to 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(America_standardized_imputed)
    inertia.append(kmeans.inertia_)

# Plot elbow method to determine optimal k
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker="o", linestyle="-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k in K-Means Clustering")
plt.show()
```

Chart: Elbow Method for Selecting k; This chart shows how inertia (sum of squared distances within clusters) decreases as k increases.

Interpretation:

* The “elbow” of the curve represents the optimal k.  
* Too few clusters: High inertia (clusters are too large and poorly defined).  
* Too many clusters: Overfitting, not useful for interpretation.

Result: We found that k \= 4 or 5 provided a good balance.

<h4 style="color:#FF5733">5. Singular Value Decomposition (SVD)
</h4>

Purpose:

* Identify which features carry the most variance (i.e., most useful for analysis).  
* Reduce dimensionality while retaining most of the information.

Code: Applying SVD

```python
from scipy.linalg import svd

# Apply SVD
U, S, Vt = svd(America_standardized_imputed)
singular_values = pd.Series(S, name="Singular Values")

# Display top singular values
print("\nTop Singular Values:\n", singular_values.head(10))
```

Key Insights:

* The first few singular values explain most of the variance.  
* If most variance is captured in the first few components, we can reduce the feature space.

<h4 style="color:#FF5733">6. Feature Correlation Analysis</h4>

Purpose:

* Identify highly correlated features (i.e., redundant features that may not add much unique information).  
* Help with feature selection and dimensionality reduction.

Code: Computing Feature Correlations

```python
# Compute correlation matrix
correlation_matrix = America_standardized_imputed.corr()

# Identify highly correlated features (absolute correlation > 0.8, excluding diagonal)
highly_correlated_features = correlation_matrix.where(
    np.triu(np.abs(correlation_matrix) > 0.8, k=1)
).stack().reset_index()
highly_correlated_features.columns = ["Feature 1", "Feature 2", "Correlation"]

print("\nHighly Correlated Features (|corr| > 0.8):\n", highly_correlated_features)
```

Key Insights:

* Features with a correlation above 0.8 are likely to be redundant.  
* Removing one of each highly correlated pair can simplify models without losing information.

Code: Visualizing Correlation Matrix

```python
# Plot heatmap of correlation matrix
plt.figure(figsize=(12, 8))
plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(ticks=range(len(correlation_matrix.columns)), labels=correlation_matrix.columns, rotation=90)
plt.yticks(ticks=range(len(correlation_matrix.index)), labels=correlation_matrix.index)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()  # Adjust layout to ensure labels fit
plt.show()
```

Chart: Feature Correlation Heatmap

This heatmap shows:

* Red areas: Highly correlated features.  
* Blue areas: Weakly correlated features.

Interpretation:

* If two features are strongly correlated, we may drop one.  
* Weakly correlated features provide independent information, which is beneficial for modeling.

Final Summary of Part 2

1\. Standardization & Missing Data Handling

* Standardized numerical features for consistent scaling.  
* Imputed missing values using column means to avoid data loss.

2\. K-Means Clustering

* Used the Elbow Method to find the optimal k (4-5 clusters).  
* Clustered respondents based on standardized features.

3\. Singular Value Decomposition (SVD)

* Identified which features carry the most variance.  
* Considered dimensionality reduction for better model performance.

4\. Feature Correlation Analysis

* Computed highly correlated features (\>0.8 correlation).  
* Plotted heatmap to visualize redundant features.


<h3 style="color:#7216e6">PART 3:</h3>


<h4 style="color:#FF5733">1. Overview of Part 3</h4>


In Part 3, we built machine learning models to predict developer salaries. The key steps included:

* Basic Linear Regression – Establishing a baseline model.  
* Feature Engineering (K-Means Clustering) – Improving prediction by incorporating cluster information.  
* Regularized Regression (Ridge) – Applying polynomial expansion and tuning regularization.  
* Non-Linear Modeling – Testing a higher-degree polynomial model.  
* Final Model Selection – Choosing the best model for prediction.  
* Each model was evaluated using cross-validation to ensure reliability.

<h4 style="color:#FF5733">2. Basic Linear Regression</h4>


Purpose:

* Establish a baseline prediction model using standardized features.  
* Use log transformation to stabilize salary variance.  
* Evaluate model performance using 5-fold cross-validation.

Code: Training a Basic Linear Model

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Prepare features and target
X_a = df.drop('CompTotal', axis=1)
y = df['CompTotal']

# Log transform target variable
y_trans = np.log1p(y)

# Standardize features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_a), columns=X_a.columns)

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_clean = pd.DataFrame(imputer.fit_transform(X_scaled), columns=X_scaled.columns)

# 5-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
basic_rms_errors = []

for train_idx, val_idx in kfold.split(X_clean):
    X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
    y_train, y_val = y_trans.iloc[train_idx], y_trans.iloc[val_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds_trans = model.predict(X_val)
    preds = np.expm1(preds_trans)  # Invert log transformation
    y_val_orig = np.expm1(y_val)

    rms = np.sqrt(mean_squared_error(y_val_orig, preds))
    basic_rms_errors.append(rms)

avg_rms_basic = np.mean(basic_rms_errors)
print(f"Basic Linear Model Average CV RMS Error: {avg_rms_basic:.4f}")
```

Key Insights:

* Baseline RMS Error Established – Provides a reference for model improvements.  
* Log Transformation Stabilizes Variance – Avoids salary distribution skewness.  
* Cross-Validation Prevents Overfitting – Ensures the model generalizes well.

<h4 style="color:#FF5733">3. Feature Engineering with K-Means Clustering</h4>


Purpose:

* Improve prediction accuracy by grouping developers into clusters.  
* Add cluster labels as categorical features to enhance model learning.

Code: K-Means Clustering for Feature Engineering

```python
from sklearn.cluster import KMeans
import pandas as pd

# Apply k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_clean)

# Convert clusters to one-hot encoded variables
cluster_dummies = pd.get_dummies(clusters, prefix="cluster")

# Combine cluster features with standardized dataset
X_b = pd.concat([X_clean, cluster_dummies], axis=1)
```

Visualization: K-Means Elbow Method

```python
inertia = []
k_values = range(2, 11)  # Testing k from 2 to 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_clean)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker="o", linestyle="-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k in K-Means Clustering")
plt.show()
```

Key Insights:

* Clustering Captures Hidden Patterns – Groups similar developers based on numeric features.  
* One-Hot Encoding Preserves Information – Prevents the loss of categorical significance.  
* Lower RMS Error – Clustering improves salary prediction accuracy.

<h4 style="color:#FF5733">4. Regularized Ridge Regression with Polynomial Expansion</h4>


Purpose:

* Prevent overfitting by applying regularization.  
* Expand feature interactions using polynomial basis functions.

Code: Ridge Regression with Polynomial Expansion

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Expand features using 2nd-degree polynomial terms
poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly2 = pd.DataFrame(poly2.fit_transform(X_clean), columns=poly2.get_feature_names_out(X_clean.columns))

# Combine polynomial features with cluster information
X_c = pd.concat([X_poly2, cluster_dummies], axis=1)
```

Key Insights:

* Polynomial Expansion Captures Non-Linear Relationships.  
* Regularization Prevents Overfitting.  
* Lower RMS Error Compared to Basic Model.

<h4 style="color:#FF5733">5. Non-Linear Data Fitting (Degree-3 Polynomial Expansion)</h4>


Purpose:

* Test a higher-order polynomial regression.  
* Evaluate whether additional feature interactions improve accuracy.

Code: Degree-3 Polynomial Model

```python
poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly3 = pd.DataFrame(poly3.fit_transform(X_clean), columns=poly3.get_feature_names_out(X_clean.columns))
```

Key Insights:

* Higher Complexity → High Variance Instability.  
* Some Improvement in Accuracy, but Risk of Overfitting.  
* Alternative Methods Needed for Stability.

Final Summary of Part 3:

* Baseline Model Established (Linear Regression).  
* Feature Engineering Improved Accuracy (K-Means).  
* Regularized Regression Balanced Accuracy & Generalization.  
* Non-Linear Model Showed Instability (Degree-3).


<h3 style="color:#7216e6">PART 4:</h3>

See PART 4 Folder for Part 4(a) and 4(b) report



















