import pytest
import os
import subprocess

def run_tests():
    """
    Run all MAVE-NN functional tests.
    
    Returns
    -------
    bool
        True if all tests pass, False otherwise
    """
    
    # Run pytest and capture the result
    result = pytest.main(['-vv', 'test_specific_tests.py'])
    
    print('test')
    
    if result == 0:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. See above for details.")

    return result == 0

