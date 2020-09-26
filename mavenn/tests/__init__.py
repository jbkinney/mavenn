from mavenn.tests.specific_tests import \
    test_GlobalEpistasisModel, \
    test_NoiseAgnosticModel, \
    test_validate_alphabet, \
    test_load, \
    test_x_to_phi_or_yhat, \
    test_load_example

def run_tests():
    """
    Run all mavenn functional tests.
    """

    test_GlobalEpistasisModel()
    test_NoiseAgnosticModel()
    test_validate_alphabet()
    test_load()
    test_x_to_phi_or_yhat()
    test_load_example()
