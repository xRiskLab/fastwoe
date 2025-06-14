"""
Compatibility tests for FastWoe across Python and scikit-learn versions.
Uses pytest with custom markers for optional execution.
"""

import os
import subprocess
import tempfile

import pytest

# Test combinations: [python_version, sklearn_version, description]
COMPATIBILITY_MATRIX = [
    ("3.9", "1.3.0", "Min supported: Python 3.9 + sklearn 1.3.0"),
    ("3.10", "1.4.2", "Python 3.10 + sklearn 1.4.x"),
    ("3.11", "1.5.2", "Python 3.11 + sklearn 1.5.x"),
    ("3.12", "latest", "Latest: Python 3.12 + latest sklearn"),
]


def run_cmd(cmd):
    """Run command and return result."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=False
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def get_numpy_constraint(python_ver, sklearn_ver):
    """Get appropriate numpy version constraint."""
    if sklearn_ver in ["1.3.0", "1.3.2"] or python_ver == "3.9":
        return "numpy<2.0"
    else:
        return "numpy>=1.21,<2.1"


@pytest.mark.compatibility
@pytest.mark.slow
@pytest.mark.parametrize("python_ver,sklearn_ver,description", COMPATIBILITY_MATRIX)
def test_python_sklearn_compatibility(python_ver, sklearn_ver, description):
    """Test FastWoe compatibility across Python and scikit-learn versions."""
    # Skip if uv not available
    success, _, _ = run_cmd("which uv")
    if not success:
        pytest.skip("uv not available - skipping compatibility tests")

    env_name = (
        f".test-py{python_ver.replace('.', '')}-sklearn{sklearn_ver.replace('.', '')}"
    )
    numpy_constraint = get_numpy_constraint(python_ver, sklearn_ver)

    # Test script content
    test_content = """
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    import fastwoe
    import sklearn
    import pandas as pd
    import numpy as np

    print(f"Testing Python {sys.version.split()[0]} + sklearn {sklearn.__version__} + numpy {np.__version__}")

    # Quick functionality test
    np.random.seed(42)
    data = pd.DataFrame({
        "cat": ["A", "B", "C"] * 10,
        "target": np.random.binomial(1, 0.3, 30)
    })

    # Test core functionality
    preprocessor = fastwoe.WoePreprocessor()
    X_proc = preprocessor.fit_transform(data[["cat"]])

    woe = fastwoe.FastWoe()
    X_woe = woe.fit_transform(X_proc, data["target"])

    # Test key methods
    mapping = woe.get_mapping("cat")
    stats = woe.get_feature_stats()
    ci = woe.predict_ci(X_proc.head(2))

    assert X_woe.shape == (30, 1), f"Expected (30, 1), got {X_woe.shape}"
    assert len(mapping) > 0, "Mapping should not be empty"
    assert len(stats) > 0, "Stats should not be empty"
    assert ci.shape[0] == 2, f"Expected 2 CI predictions, got {ci.shape[0]}"

    print("✅ All FastWoe functionality verified!")

except Exception as e:
    print(f"❌ Error: {e}")
    raise

print("SUCCESS")
"""

    try:
        # Install Python version
        success, _, stderr = run_cmd(f"uv python install {python_ver}")
        # sourcery skip: no-conditionals-in-tests
        if not success and "already installed" not in stderr.lower():
            pytest.fail(f"Failed to install Python {python_ver}: {stderr}")

        # Create environment
        run_cmd(f"rm -rf {env_name}")
        success, _, stderr = run_cmd(f"uv venv {env_name} --python {python_ver}")
        if not success:
            pytest.fail(f"Environment creation failed: {stderr}")

        python_exe = f"{env_name}/bin/python"

        # Install dependencies
        deps = [numpy_constraint, "pandas>=1.3.0", "scipy>=1.7.0"]
        if sklearn_ver == "latest":
            deps.append("scikit-learn")
        else:
            deps.append(f"scikit-learn=={sklearn_ver}")

        # sourcery skip: no-loop-in-tests
        for dep in deps:
            success, _, stderr = run_cmd(
                f"uv pip install --python {python_exe} '{dep}'"
            )
            if not success:
                pytest.fail(f"Failed to install {dep}: {stderr}")

        # Install FastWoe
        success, _, stderr = run_cmd(f"uv pip install --python {python_exe} -e .")
        if not success:
            pytest.fail(f"FastWoe installation failed: {stderr}")

        # Run test
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_content)
            test_file = f.name

        try:
            success, stdout, stderr = run_cmd(f"{python_exe} {test_file}")

            # Print output for debugging
            if stdout:
                print(f"\n--- Test Output ---\n{stdout}")

            if not success:
                pytest.fail(f"Compatibility test failed for {description}:\n{stderr}")

            # Verify success message
            assert "SUCCESS" in stdout, f"Test didn't complete successfully: {stdout}"

        finally:
            os.unlink(test_file)

    finally:
        # Cleanup
        run_cmd(f"rm -rf {env_name}")


@pytest.mark.compatibility
def test_minimum_requirements():
    """Test that minimum requirements are correctly specified."""
    # This is a lightweight test that can run without uv
    import fastwoe

    # Test that we can import everything
    assert hasattr(fastwoe, "FastWoe")
    assert hasattr(fastwoe, "WoePreprocessor")
    assert hasattr(fastwoe, "__version__")

    # Test basic instantiation
    woe = fastwoe.FastWoe()
    preprocessor = fastwoe.WoePreprocessor()

    assert woe is not None
    assert preprocessor is not None


@pytest.mark.compatibility
def test_sklearn_target_encoder_availability():
    """Test that TargetEncoder is available in the current environment."""
    try:
        from sklearn.preprocessing import TargetEncoder

        assert TargetEncoder is not None
    except ImportError:
        pytest.fail("TargetEncoder not available - need scikit-learn >= 1.3.0")


if __name__ == "__main__":
    # Allow running this file directly for development
    print("🧪 Running FastWoe compatibility tests...")
    print(
        "Use 'pytest tests/test_compatibility.py -m compatibility' for full test suite"
    )
    pytest.main([__file__, "-v", "-m", "compatibility"])
