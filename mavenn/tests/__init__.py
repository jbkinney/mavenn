from mavenn.tests.specific_tests import \
    test_GlobalEpistasisModel, \
    test_NoiseAgnosticModel, \
    test_validate_alphabet, \
    test_load, \
    test_x_to_phi_or_yhat, \
    test_GE_fit, \
    test_MPA_fit, \
    test_heatmap

def run_tests():
    """
    Run all MAVE-NN functional tests.
    """
    test_heatmap()
    test_load()
    test_validate_alphabet()
    test_x_to_phi_or_yhat()
    test_MPA_fit()
    test_GE_fit()
    test_GlobalEpistasisModel()
    test_NoiseAgnosticModel()
    
    
    
    
    

