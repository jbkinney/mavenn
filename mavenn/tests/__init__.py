from mavenn.tests.specific_tests import \
    test_GlobalEpistasisModel, \
    test_NoiseAgnosticModel, \
    test_get_1pt_variants, \
    test_validate_alphabet, \
    test_load, \
    test_x_to_phi_or_yhat, \
    test_phi_calculations, \
    test_get_additive_parameters

def run_tests():
    """
    Run all mavenn functional tests.

    parameters
    ----------
    None.

    return
    ------
    None.
    """

    test_GlobalEpistasisModel()
    test_NoiseAgnosticModel()
    test_get_1pt_variants()
    test_validate_alphabet()
    test_load()
    test_x_to_phi_or_yhat()
    test_phi_calculations()
    test_get_additive_parameters()
