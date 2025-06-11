"""MAVE-NN software package."""

# Import version
__version__ = '1.1.2'

# The functions imported here are the ONLY "maven.xxx()" functions that
# users are expected to interact with

# To regularize log calculations
import numpy as np
TINY = np.sqrt(np.finfo(np.float32).tiny)

# Primary model class
from mavenn.src.model import Model

# Examples
from mavenn.src.examples import list_tutorials
from mavenn.src.examples import run_demo
from mavenn.src.examples import load_example_dataset
from mavenn.src.examples import load_example_model

# For loading models
from mavenn.src.utils import load
from mavenn.src.utils import split_dataset

# For visualizing G-P maps
from mavenn.src.visualization import heatmap
from mavenn.src.visualization import heatmap_pairwise

# For running tests
import pytest
import os
import sys
import subprocess

def run_tests(verbose=True):
    """
    Run the MAVE-NN test suite using pytest.
    
    This function runs all tests in the mavenn/tests directory using pytest.
    It will print test results to stdout and return True if all tests pass,
    False otherwise.
    
    Parameters
    ----------
    verbose : bool, optional
        If True, prints detailed test output. If False, prints minimal output.
        Default is True.
    
    Returns
    -------
    bool
        True if all tests pass, False otherwise
    """
    # Get the path to the tests directory
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    # Configure pytest arguments based on verbosity
    pytest_args = ['pytest', test_dir]
    if verbose:
        pytest_args.extend(['-v', '--tb=short'])  # Verbose output with short tracebacks
    else:
        pytest_args.extend(['-q'])  # Quiet output
    
    # Run pytest as a subprocess to ensure output is visible
    result = subprocess.run(pytest_args)
    
    # Print a summary message
    if result.returncode == 0:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. See above for details.")
    
    # Return True if all tests passed (exit code 0), False otherwise
    return result.returncode == 0

