# FastWoe: Fast Weight of Evidence (WOE) encoding and inference

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn 1.3.0+](https://img.shields.io/badge/sklearn-1.3.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

FastWoe is a Python library for efficient **Weight of Evidence (WOE)** encoding of categorical features and statistical inference. It’s designed for machine learning practitioners seeking robust, interpretable feature engineering and likelihood-ratio-based inference for binary classification problems.

![FastWoe](https://github.com/xRiskLab/fastwoe/raw/main/ims/title.png)

## 🌟 Key Features

- **Fast WOE Encoding**: Leverages scikit-learn's `TargetEncoder` for efficient computation
- **Statistical Confidence Intervals**: Provides standard errors and confidence intervals for WOE values
- **Cardinality Control**: Built-in preprocessing to handle high-cardinality categorical features
- **Risk Differentiation Metrics**: Feature-level statistics including Gini score and Information Value (IV)
- **Compatible with scikit-learn**: Follows scikit-learn's preprocessing transformer interface
- **Statistical Foundation**: Combines Alan Turing's factor principle with Maximum Likelihood theory (see [paper](docs/woe_st_errors.md))

## 🎲 What is Weight of Evidence?

![Weight of Evidence](https://github.com/xRiskLab/fastwoe/raw/main/ims/weight_of_evidence.png)

Weight of Evidence (WOE) is a statistical technique that:

- Transforms discrete features into logarithmic scores
- Measures the strength of relationship between feature categories and true labels
- Provides interpretable coefficients as weights in logistic regression models
- Handles missing values and rare categories gracefully

**Mathematical Definition:**
```
WOE = ln(P(Event|Category) / P(Non-Event|Category)) - ln(P(Event) / P(Non-Event))
```

Where WOE represents the log-odds difference between a category and the overall population.

## 🚀 Installation

> [!IMPORTANT]  
> FastWoe requires Python 3.9+ and scikit-learn 1.3.0+ for TargetEncoder support.

### From Source
```bash
git clone https://github.com/xRiskLab/fastwoe.git
cd fastwoe
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/xRiskLab/fastwoe.git
cd fastwoe
pip install -e ".[dev]"
```

> [!TIP]
> For development work, we recommend using `uv` for faster package management:
> ```bash
> uv sync --dev
> ```

## 📖 Quick Start

```python
import pandas as pd
import numpy as np
from fastwoe import FastWoe, WoePreprocessor

# Create sample data
data = pd.DataFrame({
    'category': ['A', 'B', 'C'] * 100 + ['D'] * 50,
    'high_card_cat': [f'cat_{i}' for i in np.random.randint(0, 50, 350)],
    'target': np.random.binomial(1, 0.3, 350)
})

# Step 1: Preprocess high-cardinality features (optional)
preprocessor = WoePreprocessor(max_categories=10, min_count=5)
X_preprocessed = preprocessor.fit_transform(
    data[['category', 'high_card_cat']], 
    cat_features=['high_card_cat']  # Only preprocess this column
)

# Step 2: Apply WOE encoding
woe_encoder = FastWoe()
X_woe = woe_encoder.fit_transform(X_preprocessed, data['target'])

print("WOE-encoded features:")
print(X_woe.head())

# Step 3: Get detailed mappings with statistics
mapping = woe_encoder.get_mapping('category')
print("\nWOE Mapping for 'category':")
print(mapping[['category', 'count', 'event_rate', 'woe', 'woe_se']])
```

## 🔧 Advanced Usage

> [!CAUTION]
> When we make inferences with `predict_proba` and `predict_ci` methods, we are making a naive assumption that pieces of evidence are independent.
> The sum of WOE scores can only produce meaningful probabilistic outputs if the data is not strongly correlated among features.

### Probability Predictions

```python
# Get predictions with Naive Bayes classification
preds = woe_encoder.predict_proba(X_preprocessed)[:, 1]
print(preds.mean())
```

### Confidence Intervals

> [!NOTE]  
> Statistical confidence intervals help assess the reliability of WOE estimates, especially for categories with small sample sizes.

```python
# Get predictions with confidence intervals
ci_results = woe_encoder.predict_ci(X_preprocessed, alpha=0.05)
print(ci_results[['prediction', 'lower_ci', 'upper_ci']].head())
```

### Feature Statistics
```python
# Get comprehensive feature statistics
feature_stats = woe_encoder.get_feature_stats()
print(feature_stats[['feature', 'gini', 'information_value', 'n_categories']])
```

### Standardized WOE
```python
# Get Wald scores (standardized log-odds)
X_standardized = woe_encoder.transform_standardized(X_preprocessed, output='wald')
```

### Pipeline Integration
```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Create a complete pipeline
pipeline = Pipeline([
    ('preprocessor', WoePreprocessor(top_p=0.95, min_count=10)),
    ('woe_encoder', FastWoe()),
    ('classifier', LogisticRegression())
])

# Fit the entire pipeline
pipeline.fit(data[['category', 'high_card_cat']], data['target'])
```

## 📋 API Reference

### FastWoe Class

#### Parameters
- `encoder_kwargs` (dict): Additional parameters for sklearn's TargetEncoder
- `random_state` (int): Random state for reproducibility

#### Key Methods
- `fit(X, y)`: Fit the WOE encoder
- `transform(X)`: Transform features to WOE values
- `fit_transform(X, y)`: Fit and transform in one step
- `get_mapping(column)`: Get WOE mapping for specific column
- `predict_proba(X)`: Get probability predictions
- `predict_ci(X, alpha)`: Get predictions with confidence intervals

### WoePreprocessor Class

The `WoePreprocessor` is a preprocessing step that reduces the cardinality of categorical features. It is used to handle high-cardinality categorical features.

> [!WARNING]  
> High-cardinality features (>50 categories) can lead to overfitting and unreliable WOE estimates. Always use WoePreprocessor for such features if you plan to use in downstream tasks.

#### Parameters
- `max_categories` (int): Maximum categories to keep per feature
- `top_p` (float): Keep categories covering top_p% of frequency
- `min_count` (int): Minimum count required for category
- `other_token` (str): Token for grouping rare categories

> [!TIP]
> The `top_p` parameter uses **cumulative frequency** to select categories. For example, `top_p=0.95` keeps categories that together represent 95% of all observations, automatically grouping the long tail of rare categories into `"__other__"`. This is more adaptive than fixed `max_categories` since it preserves the most important categories regardless of their absolute count.

#### Key Methods
- `fit(X, cat_features)`: Fit preprocessor
- `transform(X)`: Apply preprocessing
- `get_reduction_summary(X)`: Get cardinality reduction statistics

**Example: Using `top_p` parameter**
```python
# Dataset with 100 categories: 
# "A" (40%), "B" (30%), "C" (15%), "D" (10%), remaining 96 categories (5% total)

preprocessor = WoePreprocessor(top_p=0.95, min_count=5)
# Result: Keeps ["A", "B", "C", "D"] (95% coverage), groups rest as "__other__"
# Reduces 100 → 5 categories while preserving 95% of the categories
```

## 📊 Theoretical Background

![A.M. Turing example](https://github.com/xRiskLab/fastwoe/raw/main/ims/turing_paper.png)

This implementation is based on rigorous statistical theory:

1. **WOE Standard Error**: `SE(WOE) = sqrt(1/n_good + 1/n_bad)`
2. **Confidence Intervals**: Using normal approximation with calculated standard errors
3. **Information Value**: Measures predictive power of each feature
4. **Gini Coefficient**: Derived from AUC to measure discriminatory power

For technical details, see [Weight of Evidence (WOE), Log Odds, and Standard Errors](docs/woe_standard_errors.md).

![Credit scoring example](https://github.com/xRiskLab/fastwoe/raw/main/ims/credit_example_woe.png)
![I.J. Good](https://github.com/xRiskLab/fastwoe/raw/main/ims/good_bayes_odds.png)

## 🧪 Testing

Run the test suite:
```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=fastwoe --cov-report=html
```

## 🛠️ Development

### Development Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/xRiskLab/fastwoe.git
cd fastwoe
uv sync --dev
```

### Running Tests

Run the main test suite:
```bash
uv run pytest
```

Run tests without slow compatibility tests:
```bash
uv run pytest -m "not slow"  
```

Run compatibility tests across Python/scikit-learn versions (requires `uv`):
```bash
uv run pytest -m compatibility
```

Run specific test categories:
```bash
# Only fast compatibility checks
uv run pytest -m "compatibility and not slow"

# Only slow cross-version tests  
uv run pytest -m "compatibility and slow"
```

### Building the Package

Build wheel and source distribution:
```bash
uv build
```

Install from local build:
```bash
uv pip install dist/fastwoe-*.whl
```

Test installation in clean environment:
```bash
# Create temporary environment  
uv venv .test-env --python 3.9
uv pip install --python .test-env/bin/python dist/fastwoe-*.whl
.test-env/bin/python -c "import fastwoe; print(f'FastWoe {fastwoe.__version__} installed successfully!')"
```

### Code Quality

Format code:
```bash
uv run black fastwoe/ tests/
```

Lint code:
```bash
uv run ruff check fastwoe/ tests/
```

## 📈 Performance Characteristics

- **Memory Efficient**: Uses pandas and numpy for vectorized operations
- **Scalable**: Handles datasets with millions of rows
- **Fast**: Leverages sklearn's optimized TargetEncoder implementation
- **Robust**: Handles edge cases like single categories and missing values

## 📋 Changelog

### Version 0.1.0 (Current)

**Initial Release** 🎉

#### ✨ Features
- **Core WOE Implementation**: Fast Weight of Evidence encoding using scikit-learn's TargetEncoder
- **Statistical Rigor**: MLE-based standard errors and confidence intervals for WOE estimates
- **High-Cardinality Support**: WoePreprocessor for handling features with many categories
- **Comprehensive Statistics**: Gini coefficient, Information Value (IV), and feature-level metrics
- **Integration with scikit-learn**: Full compatibility with sklearn pipelines and transformers
- **Cross-Version Testing**: Compatibility verified across Python 3.9-3.12 and sklearn 1.3.0+

#### 🔧 Technical
- **Build System**: Modern `pyproject.toml` with Hatchling backend
- **Testing**: 26+ comprehensive tests with 94% code coverage
- **Documentation**: Complete API reference and mathematical background
- **Examples**: Jupyter notebooks and practical usage examples

#### 📊 Supported Operations
- `fit()`, `transform()`, `fit_transform()`: Core WOE encoding
- `get_mapping()`: Detailed category-level WOE mappings
- `predict_ci()`: Predictions with confidence intervals
- `get_feature_stats()`: Feature-level discrimination metrics
- `transform_standardized()`: Wald scores and standardized outputs

> [!NOTE]  
> This is a beta release. The API is not considered stable for production use.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. Alan M. Turing (1942). The Applications of Probability to Cryptography.
2. I. J. Good (1950). Probability and the Weighing of Evidence.
3. Daniele Micci-Barreca (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems.
4. Naeem Siddiqi (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.

## 🔗 Other Projects

- [scikit-learn](https://scikit-learn.org/): Python Machine learning library providing TargetEncoder implementation
- [category_encoders](https://contrib.scikit-learn.org/category_encoders/): Additional categorical encoding methods
- [WoeBoost](https://github.com/xRiskLab/woeboost): Weight of Evidence (WOE) Gradient Boosting in Python

## ℹ️ Additional Information

- **Documentation**: [README.md](README.md) and [Theoretical Background](docs/woe_standard_errors.md)
- **Examples**: See [examples/](examples/) directory