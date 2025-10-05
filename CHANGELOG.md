# Changelog

## Version 0.1.4.post2 (Current)

**Monotonic Constraints Support**: Complete implementation across all binning methods

- **New Features**:
  - **Monotonic Constraints**: Added comprehensive monotonic constraints support for credit scoring compliance
    - **Tree Method**: Native scikit-learn monotonic constraints (`monotonic_cst` parameter)
    - **KBins Method**: Isotonic regression post-processing to enforce constraints
    - **FAISS KMeans Method**: Isotonic regression post-processing to enforce constraints
    - **Constraint Values**: `1` (increasing), `-1` (decreasing), `0` (no constraint)
    - **Validation**: Comprehensive input validation with clear error messages
    - **Binning Info**: Monotonic constraints stored in `binning_info_` and displayed in summaries
  - **Isotonic Regression Integration**: Added `_apply_isotonic_constraints` method
    - Uses scikit-learn's `IsotonicRegression` for KBins and FAISS methods
    - Enforces monotonic patterns on WOE values after initial binning
    - Handles bin center extraction and constraint application
  - **Comprehensive Testing**: Added extensive test coverage for monotonic constraints
    - Tests for all binning methods (Tree, KBins, FAISS)
    - Tests for multiclass and continuous targets
    - Tests for edge cases and validation
    - Tests for backward compatibility
  - **Enhanced Documentation**: Updated README and examples
    - Added detailed monotonic constraints section with examples
    - Updated API reference with `monotonic_cst` parameter
    - Created comprehensive example (`examples/fastwoe_monotonic.py`)

- **API Changes**:
  - **FastWoe**: Added `monotonic_cst` parameter to constructor
    - Type: `dict[str, int]` mapping feature names to constraint values
    - Default: `None` (no constraints applied)
    - Validation: Ensures valid constraint values (-1, 0, 1) and feature names
  - **Binning Summary**: Added `monotonic_constraint` column to `get_binning_summary()`
  - **Binning Info**: Added `monotonic_constraint` field to `binning_info_`

- **Bug Fixes**:
  - Fixed bare `except:` clause in isotonic constraints implementation
  - Updated test for unsupported methods warning (now tests invalid method instead)
  - Fixed FAISS API compatibility issue with `index.search` method

- **Examples**:
  - **New**: `examples/fastwoe_monotonic.py` - Comprehensive monotonic constraints demonstration
    - Shows all binning methods with constraints
    - Compares KBins strategies (uniform, quantile, kmeans)
    - Analyzes monotonic patterns and performance
    - Provides readable table outputs instead of plots

- **Technical Details**:
  - **Tree Method**: Uses scikit-learn's native `monotonic_cst` parameter
  - **KBins/FAISS Methods**: Applies isotonic regression after WOE calculation
  - **Performance**: Constraints may slightly affect performance but ensure business logic compliance
  - **Compatibility**: Fully backward compatible - existing code works unchanged

## Version 0.1.4.post1

**Bug Fix Release**: Fixed pyrefly type checking comments appearing in output

- **Bug Fixes**:
  - Fixed `# pyrefly: ignore` comments being printed in `WeightOfEvidence.summary()` output
  - Migrated from `pyrefly` to `ty` type checker
  - Updated all type checking comments to use standard `# type: ignore[error-code]` format
  - Fixed f-string formatting issues in summary method

- **Infrastructure**:
  - **Type Checking Migration**: Complete migration from `pyrefly` to `ty`
    - Moved `ty.toml` configuration into `pyproject.toml`
    - Updated GitHub Actions workflow to use `ty`
    - Updated Makefile targets for `ty` commands
    - Updated documentation for new type checking setup
  - **Dependencies**: Updated dev dependencies to use `ty>=0.0.1a21`

## Version 0.1.4 (Current)

**Multiclass Support & Enhanced Tree Binning**: Major feature additions and API improvements

- **New Features**:
  - **Multiclass WOE Support**: Added one-vs-rest Weight of Evidence encoding for multiclass targets
    - Automatic detection of multiclass targets (3+ unique values, not continuous proportions)
    - One-vs-rest binary encoding for each class against all others
    - Multiple output columns per feature: `feature_class_0`, `feature_class_1`, etc.
    - Support for both integer and string class labels
    - Class-specific priors stored in `y_prior_` dictionary
  - **Enhanced Tree Binning**: Improved decision tree-based numerical feature binning
    - Fixed NaN values in last bin issue with proper right-inclusive binning `(a, b]`
    - Added `get_tree_estimator(feature)` method to access underlying scikit-learn trees
    - Optimized default parameters for credit scoring: `max_depth=3`, `random_state=42`
    - Simplified default tree parameters (removed `min_samples_leaf`, `min_samples_split`)
  - **Unified Binner Parameters**: Streamlined API with single `binner_kwargs` parameter
    - Replaced separate `tree_kwargs` and `faiss_kwargs` with unified approach
    - Backward compatibility maintained for existing parameter names
    - Cleaner API: `FastWoe(binning_method="tree", binner_kwargs={"max_depth": 2})`

- **API Changes**:
  - **Default Binning Method**: Changed from `"kbins"` to `"tree"` for numerical features
  - **New Method**: `get_tree_estimator(feature)` to access fitted decision tree estimators
  - **Enhanced Target Detection**: Automatic multiclass detection with `is_multiclass_target` attribute
  - **Class Information**: Added `classes_` and `n_classes_` attributes for multiclass targets

- **Fixed**:
  - **Tree Binning NaN Bug**: Resolved issue where last bin always contained NaN values
  - **Binning Logic**: Implemented proper right-inclusive binning `(a, b]` instead of `np.digitize`
  - **Split Point Handling**: Improved `_create_bin_edges_from_splits` to handle duplicate splits
  - **Test Coverage**: Added comprehensive tests for multiclass and tree binning edge cases

- **Documentation & Examples**:
  - **New Example**: `examples/fastwoe_multiclass.py` demonstrating multiclass WOE usage
  - **Comprehensive Tests**: Added `TestMulticlassWoe` class with 9 test methods
  - **Updated Documentation**: Clarified multiclass WOE concept and usage patterns

- **Performance & Reliability**:
  - **Credit Scoring Optimization**: Default tree parameters optimized for 4-8 bins per feature
  - **Reproducible Results**: `random_state=42` as default for consistent binning
  - **Memory Efficiency**: Improved handling of multiclass target encoding
  - **Error Handling**: Enhanced validation for multiclass target types

## Version 0.1.3.post1

**Enhanced Statistical Analysis**: Added IV standard errors and Series support

- **New Features**:
  - **IV Standard Errors**: Added statistical rigor to Information Value calculations
    - `get_iv_analysis()` method for detailed IV analysis with confidence intervals
    - IV standard errors calculated using delta method with WOE variance propagation
    - Statistical significance testing for IV values (confidence intervals)
    - Enhanced `feature_stats_` with `iv_se`, `iv_ci_lower`, `iv_ci_upper` fields
  - **Mathematical Framework**: Implements proper uncertainty quantification for IV
    - Delta method for variance propagation: `Var(IV) ≈ Σ (bad_rate - good_rate)² * Var(WOE)`
    - Sampling variance corrections for rate differences
    - Normal approximation confidence intervals with lower bound ≥ 0

- **Fixed**:
  - **Series Input Support**: Resolved `'Series' object has no attribute 'columns'` error when passing Series to `fit()`/`transform()`
  - **Enhanced Input Handling**: Both methods now accept `pd.Series`, `pd.DataFrame`, and `np.ndarray` uniformly
  - **Automatic Conversion**: Series converted to single-column DataFrames with appropriate column names
  - **Updated Type Hints**: Method signatures now include `pd.Series` support

- **API Enhancements**:
  - **New Method**: `get_iv_analysis(col=None, alpha=0.05)` for comprehensive IV statistics
  - **Enhanced Feature Stats**: All feature statistics now include IV uncertainty measures
  - **Statistical Significance**: Automatic classification of IV significance based on confidence intervals

- **User Experience Improvements**:
  - **Silent Numerical Binning**: Changed default `warn_on_numerical=False` since numerical feature binning is now a core feature
  - **Cleaner Output**: No warnings by default for automatic binning (KBinsDiscretizer, FAISS KMeans, Decision Tree)
  - **Optional Warnings**: Users can still enable warnings with `warn_on_numerical=True` if desired

## Version 0.1.3

**Stable Release**: Enhanced input handling and Series support

- **Major Improvements**:
  - **Series Input Support**: `FastWoe.fit()` and `FastWoe.transform()` now accept `pd.Series` inputs
  - **Automatic Conversion**: Series are automatically converted to single-column DataFrames with appropriate column names
  - **Enhanced Type Hints**: Updated method signatures to include `pd.Series` in Union types
  - **Comprehensive Warnings**: Informative warnings guide users about automatic conversions

- **Bug Fixes**:
  - **Fixed Series AttributeError**: Resolved `'Series' object has no attribute 'columns'` error when passing Series to fit/transform
  - **Consistent API**: Both fit and transform methods now handle Series, DataFrame, and numpy array inputs uniformly

- **Testing**:
  - All existing tests continue to pass
  - Added comprehensive Series support verification
  - Verified compatibility with named and unnamed Series
  - Tested numerical Series with automatic binning

## Version 0.1.2a

**Alpha Release**: Stable dependency resolution and full Python 3.9-3.12 + NumPy 2.0 support

- **Major Improvements**:
  - **Simplified Dependencies**: Unified scikit-learn constraint `>=1.3.0,<1.7.0` (resolves to 1.6.1) across all Python versions
  - **Universal Compatibility**: Works seamlessly with Python 3.9-3.12 and NumPy 2.0
  - **Resolved Dependency Conflicts**: Eliminated `uv sync --dev` resolution errors
  - **NumPy 2.0 Support**: Full compatibility without ComplexWarning import issues
  - **FAISS Integration**: Verified working with latest FAISS + NumPy 2.0
  - **Bug Fix**: Added proper NumPy array support in `FastWoe.transform()` method

- **Testing**:
  - All 107 core tests pass
  - All 4 compatibility tests pass
  - Verified across Python 3.9 (NumPy 1.x) and Python 3.12 (NumPy 2.0)

## Version 0.1.2.post12

- **Fixed**:
  - **Compatibility Test Versions**: Updated to use scikit-learn 1.3.2 for Python 3.9 and scikit-learn 1.7.2 for Python 3.12
  - **NumPy 2.0 Support**: scikit-learn 1.7.2 properly supports NumPy 2.0 without ComplexWarning import errors
  - **Dependency Installation**: Fixed NumPy constraint installation to ensure correct versions are installed
  - **Test Matrix**: Simplified compatibility test matrix to focus on critical combinations

## Version 0.1.2.post11

- **Fixed**:
  - **CI Workflow Separation**: Separated main CI tests from compatibility tests to prevent failures
  - **Main CI**: Now skips compatibility tests with `-m "not compatibility"` to avoid scikit-learn/NumPy conflicts
  - **Compatibility Workflow**: Updated to use scikit-learn 1.5.2 for Python 3.12 + NumPy 2.0 support
  - **Test Organization**: Main CI runs 107 core tests, compatibility workflow runs 4 compatibility tests separately

## Version 0.1.2.post10

- **Fixed**:
  - **CI Compatibility Test**: Fixed dependency installation order and added version debugging
  - **scikit-learn Version**: Ensured scikit-learn 1.4.2 is correctly installed for NumPy 2.0 compatibility
  - **Installation Process**: Improved dependency resolution by installing all packages at once

## Version 0.1.2.post9

- **Fixed**:
  - **CI Compatibility Test**: Fixed Python 3.12 + NumPy 2.0 compatibility test by using scikit-learn 1.4.2 instead of latest
  - **ComplexWarning Import Error**: Resolved scikit-learn import error with NumPy 2.0 by pinning to compatible version
  - **Test Matrix Update**: Updated compatibility test to use scikit-learn 1.4.2 which supports NumPy 2.0

## Version 0.1.2.post8

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
