# Changelog

## Version 0.1.2.post8 (Current)

- **Fixed**:
  - **NumPy 2.x Compatibility**: Removed `numpy<2.0` constraint to allow NumPy 2.x installation
  - **FAISS NumPy 2.x Support**: Updated FAISS dependency to `>=1.12.0` for full NumPy 2.x compatibility
  - **Dependency Resolution**: Fixed dependency conflicts when using `uv add "numpy>2.0.0"` with `fastwoe[faiss]`

## Version 0.1.2.post7

- **Fixed**:
  - **Python 3.12 Support**: Updated `faiss-cpu` dependency to `>=1.12.0` to support Python 3.12
  - **Compatibility Test**: Fixed failing compatibility test for Python 3.12 + latest sklearn

## Version 0.1.2.post6

- **Fixed**:
  - **FAISS Import Path**: Fixed `AttributeError: module 'faiss' has no attribute 'KMeans'` by using correct import path `faiss.extra_wrappers.Kmeans`
  - **FAISS NumPy Compatibility**: Updated `faiss-cpu` dependency from `>=1.7.0` to `>=1.12.0` for Python 3.12 support and better NumPy 2.x compatibility
- **Added**:
  - **Comprehensive FAISS Documentation**: Added troubleshooting section with solutions for common FAISS import errors
  - **Compatibility Notes**: Clear guidance on FAISS version requirements for different NumPy versions
  - **Verification Code**: Added test snippet for users to verify FAISS installation

## Version 0.1.2.post5

- **Fixed**:
  - **Numpy Array Support in Transform**: Fixed `AttributeError` when passing numpy arrays to `FastWoe.transform()`
    - Added automatic conversion of numpy arrays to pandas DataFrames with generic column names
    - Consistent behavior with `fit()` method which already supported numpy arrays
    - Added appropriate warning messages for users about column name conversion
    - Resolves issue where `encoder.transform(x_grid.reshape(-1, 1))` would crash with `'numpy.ndarray' object has no attribute 'index'`
- **Added**:
  - **Test Coverage**: Added comprehensive test cases for numpy array input to `transform()` method
    - Tests both single and multi-feature numpy arrays
    - Verifies proper conversion and output format
    - Ensures no more AttributeError crashes

## Version 0.1.2.post4

- **Added**:
  - **FAISS KMeans Binning**: New `faiss_kmeans` binning method for numerical features using FAISS clustering
    - Efficient clustering-based binning using Facebook's FAISS library
    - Optional dependency: install with `pip install fastwoe[faiss]`
    - Configurable parameters: `k` (clusters), `niter` (iterations), `verbose`, `gpu`
    - Creates meaningful bin labels based on cluster centroids
    - Handles missing values appropriately
    - GPU acceleration support for large datasets
  - **Comprehensive Testing**: Added 9 test cases covering all FAISS KMeans functionality
  - **Example Scripts**: Created `examples/fastwoe_faiss_kmeans.py` demonstrating usage and performance comparison
- **Improved**:
  - Enhanced `FastWoe` constructor with `faiss_kwargs` parameter for FAISS configuration
  - Updated `pyproject.toml` with optional FAISS dependency group
  - Consistent API integration with existing binning methods (`kbins`, `tree`, `faiss_kmeans`)

## Version 0.1.2.post3

- **Fixed**:
  - Resolved the continuous target calculation compatibility issue when calculating bin statistics and metrics.

## Version 0.1.2.post2

- **Fixed**:
  - Tree binning with continuous targets: Fixed `ValueError` when using `binning_method='tree'` with continuous target values (proportions). The code now automatically selects `DecisionTreeRegressor` for continuous targets and `DecisionTreeClassifier` for binary targets.
- **Improved**:
  - Enhanced tree binning logic to handle both classification and regression scenarios
  - Better target type detection for optimal tree estimator selection

##  Version 0.1.2.post1
- **Fixed**:
  - **Continuous target**: Allowed a case where the target is continuous.

## Version 0.1.2
- **Improved**:
  - Added decision tree-based binning (`DecisionTreeClassifier`) for numerical features
  - Implemented fast Somers' D in numba for non-binary target variables (e.g., loss rates)
    - Fastest Somers' D implementation in Python (3x faster than scipy)
    - Produces both `D_Y|X` and `D_X|Y` scores

### Version 0.1.2

**Decision Tree-Based Binning** 🌳

#### ✨ New Features
- **Tree-Based Binning**: Added `binning_method="tree"` for intelligent, target-aware numerical feature binning
- **Flexible Configuration**: `tree_kwargs` parameter for customizing tree hyperparameters
- **Automatic Bin Discovery**: Number of bins determined by tree structure rather than fixed structures

#### 🔧 API Enhancements
- **New Parameters**:
  - `binning_method`: Choose between "kbins" (default) and "tree" methods
  - `tree_estimator`: Tree estimator class
  - `tree_kwargs`: Dictionary of tree parameters
- **Enhanced Binning Summary**: Added "method" column to show which binning method was used

#### 📊 Benefits of Tree Binning
- **Target-Aware Splits**: Bins optimized for the target variable relationship
- **Non-Linear Pattern Capture**: Better handling of complex numerical relationships
- **Adaptive Bin Count**: Automatically determines optimal number of bins
- **Backward Compatibility**: All existing functionality preserved

#### 🎯 Usage Examples
```python
# Traditional binning (unchanged)
woe_encoder = FastWoe(binning_method="kbins")

# Tree-based binning
woe_encoder = FastWoe(
    binning_method="tree",
    tree_kwargs={"max_depth": 3, "min_samples_split": 20}
)
```

## Version 0.1.1.post3

- **Fixed**:
  - **sklearn Compatibility**: Simplified sklearn compatibility by always using `quantile_method="averaged_inverted_cdf"` parameter since FastWoe requires `scikit-learn>=1.3.0` where this parameter is always supported.

- **Improved**:
  - Removed unnecessary sklearn version detection code
  - Always applies `quantile_method="averaged_inverted_cdf"` for consistent binning behavior
  - Maintains `scikit-learn>=1.3.0` requirement

## Version 0.1.2.post1

- **Fixed**:
  - GitHub workflow compatibility: Fixed `TypeError` with `quantile_method` parameter in `KBinsDiscretizer` for older scikit-learn versions. The code now properly checks scikit-learn version (>= 1.7.0) before using the `quantile_method` parameter.
  - Added explicit `packaging>=21.0` dependency for reliable version checking
- **Improved**:
  - Enhanced version compatibility checking with proper fallback mechanisms
  - All GitHub workflow tests now pass successfully

## Version 0.1.1.post3

- **Fixed**:
  - sklearn version compatibility: Fixed `TypeError` with `quantile_method` parameter in `KBinsDiscretizer` for older sklearn versions (< 1.3.0). The code now checks sklearn version and only uses `quantile_method` when supported.
  - API consistency: `predict_ci()` method now returns a numpy array instead of ald  DataFrame, consistent with `predict_proba()`. Returns shape `(n_samples, 2)` with columns `[ci_lower, ci_upper]`.
- **Improved**:
  - Added comprehensive tests to verify compatibility across different sklearn versions
  - Updated `WeightOfEvidence` interpretability module to work with the new `predict_ci` format
- **Notes**:
  - All changes from `0.1.1.post2` are included in this release.
  - This release supersedes `0.1.1.post2`.

## Version 0.1.1.post2

- **Fixed**:
  - NumPy array input handling: `FastWoe.fit` and related methods now accept NumPy arrays as input, automatically converting them to pandas DataFrames/Series with a warning. This prevents `AttributeError` and improves user experience.
- **Notes**:
  - All changes from `0.1.1.post1` are included in this release.
  - This release supersedes `0.1.1.post1`.

## Version 0.1.1.post1

- **Bug Fixes**:
  - Fixed issues with pandas/numpy data type conversions
  - Improved handling of rare categories in WOE calculations
  - Better error messages for edge cases

## Version 0.1.1

**Enhanced Interpretability Module** 🚀

### ✨ New Features
- **WeightOfEvidence Interpretability**: Explanation module for FastWoe classifiers
- **Auto-Inference Capabilities**: Automatically detect and infer feature names, class names, and training data
- **Unified Explanation API**: Single `explain()` method supporting both single samples and dataset+index patterns
- **Enhanced Output Control**: `return_dict` parameter for clean formatted output vs dictionary return

### 🔧 Usability Improvements
- **Flexible Input Handling**: Support for numpy arrays, pandas Series/DataFrames, and mixed data types
- **Consistent Class Formatting**: Unified formatting between true labels and predicted classes
- **Enhanced Examples**: Comprehensive examples showing FastWoe vs traditional classifiers

### 📊 Enhanced API
- `WeightOfEvidence()`: Auto-inference factory with intelligent parameter detection
- `explain(sample)` and `explain(dataset, sample_idx)`: Dual usage patterns for maximum flexibility
- `explain_ci(sample, alpha=0.05)`: Explain with confidence intervals for uncertainty quantification

## Version 0.1.0

**Initial Release** 🎉

### ✨ Features
- **Core WOE Implementation**: Fast Weight of Evidence encoding using scikit-learn's TargetEncoder
- **Statistical Rigor**: MLE-based standard errors and confidence intervals for WOE estimates
- **High-Cardinality Support**: WoePreprocessor for handling features with many categories
- **Comprehensive Statistics**: Gini coefficient, Information Value (IV), and feature-level metrics
- **Integration with scikit-learn**: Full compatibility with sklearn pipelines and transformers
- **Cross-Version Testing**: Compatibility verified across Python 3.9-3.12 and sklearn 1.3.0+

### 📊 Supported Operations
- `fit()`, `transform()`, `fit_transform()`: Core WOE encoding
- `get_mapping()`: Detailed category-level WOE mappings
- `predict_ci()`: Predictions with confidence intervals
- `get_feature_stats()`: Feature-level discrimination metrics
- `transform_standardized()`: Wald scores and standardized outputs
