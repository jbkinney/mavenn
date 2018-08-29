Testing load_dataset() parameter file_arg ...
Test # 0: Error, as expected:  Not all sequences are the same length.
Test # 1: Error, as expected:  Counts must be nonnegative numbers.
Test # 2: 0
Error, as expected:  Invalid character found in sequences.
Test # 3: Error, as expected:  Noninteger numbers found in counts.
Test # 4: Error, as expected:  Invalid column in dataframe: x
Test # 5: Success, as expected.
Test # 6: Success, as expected.
Test # 7: Success, as expected.
Test # 8: Success, as expected.
Test # 9: Success, as expected.
Test # 10: Success, as expected.
Test # 11: Success, as expected.
Test # 12: Success, as expected.
Tests passed: 13. Tests failed: 0.

Testing load_dataset() parameter file_arg ...
Test # 13: 0
Error, as expected:  Invalid character found in sequences.
Test # 14: Success, as expected.
Test # 15: Success, as expected.
Tests passed: 16. Tests failed: 0.

Testing load_dataset() parameter file_arg ...
Test # 16: 0
Error, as expected:  Invalid character found in sequences.
Test # 17: Success, as expected.
Tests passed: 18. Tests failed: 0.

Testing load_model() parameter file_arg ...
Test # 18: Error, as expected:  Dataframe represents model with invalid columns: ['val_A', 'val_C', 'val_G', 'val_X']
Test # 19: Error, as expected:  Positions cannot be interpreted as integers.
Test # 20: Success, as expected.
Tests passed: 21. Tests failed: 0.

Testing SimulateLibrary() without parameters.
Test # 21: Success, as expected.
Testing SimulateLibrary() parameter wtseq ...
Test # 22: Error, as expected:  type(wtseq) = <class 'int'>; must be a string 
Test # 23: Error, as expected:  type(wtseq) = <class 'float'>; must be a string 
Test # 24: Error, as expected:  wtseq can only contain bases in dict_keys(['A', 'C', 'G', 'T'])
Test # 25: Error, as expected:  type(wtseq) = <class 'bool'>; must be a string 
Test # 26: Error, as expected:  wtseq length cannot be 0
Test # 27: Success, as expected.
Test # 28: Success, as expected.
Tests passed: 29. Tests failed: 0.

Testing SimulateLibrary() parameter mutrate ...
Test # 29: Error, as expected:  mutrate = 1; must be 0 <= mutrate <= 1.
Test # 30: Error, as expected:  type(mutrate) = <class 'int'>; must be a float 
Test # 31: Error, as expected:  type(mutrate) = <class 'int'>; must be a float 
Test # 32: Error, as expected:  type(mutrate) = <class 'int'>; must be a float 
Test # 33: Success, as expected.
Test # 34: Success, as expected.
Tests passed: 35. Tests failed: 0.

Testing SimulateLibrary() parameter numseq ...
Test # 35: Error, as expected:  type(numseq) = <class 'str'>; must be a int 
Test # 36: Error, as expected:  numseq = -1 must be a positive int 
Test # 37: Error, as expected:  numseq = 0 must be a positive int 
Test # 38: Error, as expected:  type(numseq) = <class 'float'>; must be a int 
Test # 39: Success, as expected.
Test # 40: Success, as expected.
Test # 41: Success, as expected.
Test # 42: Success, as expected.
Tests passed: 43. Tests failed: 0.

Testing SimulateLibrary() parameter dicttype ...
Test # 43: Error, as expected:   'dicttype' must be either 'dna', 'rna', or 'protein', entered dicttype: x 
Test # 44: Error, as expected:   'dicttype' must be either 'dna', 'rna', or 'protein', entered dicttype: 1 
Test # 45: Error, as expected:   'dicttype' must be either 'dna', 'rna', or 'protein', entered dicttype: True 
Test # 46: Success, as expected.
Test # 47: Success, as expected.
Test # 48: Success, as expected.
Tests passed: 49. Tests failed: 0.

Testing SimulateLibrary() parameter probarr ...
Test # 49: Error, as expected:  type(probarr) = <class 'int'>; must be an np.ndarray 
Test # 50: Error, as expected:  type(probarr) = <class 'float'>; must be an np.ndarray 
Test # 51: Error, as expected:  type(probarr) = <class 'str'>; must be an np.ndarray 
Test # 52: Error, as expected:  type(probarr) = <class 'list'>; must be an np.ndarray 
Test # 53: Success, as expected.
Tests passed: 54. Tests failed: 0.

Testing SimulateLibrary() parameter tags ...
Test # 54: Error, as expected:  type(tags) = <class 'NoneType'>; must be an boolean 
Test # 55: Error, as expected:  type(tags) = <class 'int'>; must be an boolean 
Test # 56: Error, as expected:  type(tags) = <class 'float'>; must be an boolean 
Test # 57: Success, as expected.
Test # 58: Success, as expected.
Tests passed: 59. Tests failed: 0.

Testing SimulateLibrary() parameter tag_length ...
Test # 59: Error, as expected:  type(tag_length) = <class 'NoneType'>; must be an int 
Test # 60: Error, as expected:  tag_length = -1 must be a positive int 
Test # 61: Error, as expected:  type(tag_length) = <class 'float'>; must be an int 
Test # 62: Success, as expected.
Test # 63: Success, as expected.
Tests passed: 64. Tests failed: 0.

Error:  Not all sequences are the same length.
Testing SimulateSort() parameter df ...
Test # 64: Error, as expected:  type(df) = <class 'int'>; must be a pandas dataframe 
Test # 65: Error, as expected:   Simulate Sort Requires pandas dataframe as input dataframe. Entered df was 'None'.
Test # 66: Error, as expected:  Invalid column in dataframe: pos
Test # 67: Success, as expected.
Test # 68: Success, as expected.
Test # 69: Success, as expected.
Test # 70: Success, as expected.
Tests passed: 71. Tests failed: 0.

Testing SimulateSort() parameter mp ...
Test # 71: Error, as expected:   Simulate Sort Requires pandas dataframe as model input. Entered model df was 'None'.
Test # 72: Error, as expected:  type(mp) = <class 'str'>; must be a pandas dataframe 
Test # 73: Success, as expected.
Test # 74: Success, as expected.
Tests passed: 75. Tests failed: 0.

Testing SimulateSort() parameter noisetype ...
Test # 75: Error, as expected:  type(noisetype) = <class 'int'>; must be a string 
Test # 76: Error, as expected:  type(noisetype) = <class 'float'>; must be a string 
Test # 77: Error, as expected:  noisetype = x; must be in ['LogNormal', 'Normal', 'None', 'Plasmid']
Test # 78: Error, as expected:  For a LogNormal noise model there must 
                         be 2 input parameters
Test # 79: Success, as expected.
Test # 80: Success, as expected.
Tests passed: 81. Tests failed: 0.

Testing SimulateSort() parameter nbins ...
Test # 81: Error, as expected:  type(nbins) = <class 'str'>; must be of type int 
Test # 82: Error, as expected:  number of bins must be greater than 1, entered bins = -1
Test # 83: Error, as expected:  type(nbins) = <class 'float'>; must be of type int 
Test # 84: Error, as expected:  number of bins must be greater than 1, entered bins = 1
Test # 85: Success, as expected.
Test # 86: Success, as expected.
Test # 87: Success, as expected.
Tests passed: 88. Tests failed: 0.

Error:  Not all sequences are the same length.
Testing ProfileFreq() parameter dataset_df ...
Test # 88: Error, as expected:  type(df) = <class 'int'>; must be a pandas dataframe 
Test # 89: Error, as expected:  type(df) = <class 'str'>; must be a pandas dataframe 
Test # 90: Error, as expected:   Profile freq requires pandas dataframe as input dataframe. Entered df was 'None'.
Test # 91: Error, as expected:   Profile freq requires pandas dataframe as input dataframe. Entered df was 'None'.
Test # 92: Success, as expected.
Test # 93: Success, as expected.
Tests passed: 94. Tests failed: 0.

Testing ProfileFreq() parameter bin ...
Test # 94: Error, as expected:  bin = -1 must be a positive int 
Test # 95: Error, as expected:  type(bin) = <class 'str'>; must be of type int 
Test # 96: Error, as expected:  type(bin) = <class 'float'>; must be of type int 
Test # 97: Success, as expected.
Test # 98: Success, as expected.
Tests passed: 99. Tests failed: 0.

Testing ProfileFreq() parameter start ...
Test # 99: Error, as expected:  type(start) = <class 'float'>; must be of type int 
Test # 100: Error, as expected:  type(start) = <class 'str'>; must be of type int 
Test # 101: Error, as expected:  type(start) = <class 'float'>; must be of type int 
Test # 102: Error, as expected:  type(start) = <class 'NoneType'>; must be of type int 
Test # 103: Success, as expected.
Test # 104: Success, as expected.
Test # 105: Success, as expected.
Test # 106: Success, as expected.
Tests passed: 107. Tests failed: 0.

Testing ProfileFreq() parameter end ...
Test # 107: Error, as expected:  type(end) = <class 'float'>; must be of type int 
Test # 108: Error, as expected:  type(end) = <class 'str'>; must be of type int 
Test # 109: Error, as expected:  type(end) = <class 'float'>; must be of type int 
Test # 110: Success, as expected.
Test # 111: Success, as expected.
Test # 112: Success, as expected.
Test # 113: Success, as expected.
Tests passed: 114. Tests failed: 0.

Error:  Not all sequences are the same length.
Testing ProfileMut() parameter dataset_df ...
Test # 114: Error, as expected:  type(df) = <class 'int'>; must be a pandas dataframe 
Test # 115: Error, as expected:  type(df) = <class 'str'>; must be a pandas dataframe 
Test # 116: Error, as expected:   Profile info requires pandas dataframe as input dataframe. Entered df was 'None'.
Test # 117: Error, as expected:   Profile info requires pandas dataframe as input dataframe. Entered df was 'None'.
Test # 118: Success, as expected.
Test # 119: Success, as expected.
Tests passed: 120. Tests failed: 0.

Testing ProfileMut() parameter bin ...
Test # 120: Error, as expected:  Column "ct_-1" is not in columns=Index(['ct', 'ct_0', 'ct_1', 'ct_2', 'ct_3', 'ct_4', 'ct_5', 'ct_6', 'ct_7',
       'ct_8', 'ct_9', 'seq'],
      dtype='object')
Test # 121: Error, as expected:  type(bin) = <class 'str'>; must be a int 
Test # 122: Error, as expected:  type(bin) = <class 'float'>; must be a int 
Test # 123: Success, as expected.
Test # 124: Success, as expected.
Tests passed: 125. Tests failed: 0.

Testing ProfileMut() parameter start ...
Test # 125: Error, as expected:  type(start) = <class 'float'>; must be of type int 
Test # 126: Error, as expected:  type(start) = <class 'str'>; must be of type int 
Test # 127: Error, as expected:  type(start) = <class 'float'>; must be of type int 
Test # 128: Error, as expected:  type(start) = <class 'NoneType'>; must be of type int 
Test # 129: Success, as expected.
Test # 130: Success, as expected.
Test # 131: Success, as expected.
Test # 132: Success, as expected.
Tests passed: 133. Tests failed: 0.

Testing ProfileMut() parameter end ...
Test # 133: Error, as expected:  type(end) = <class 'float'>; must be of type int 
Test # 134: Error, as expected:  type(end) = <class 'str'>; must be of type int 
Test # 135: Error, as expected:  type(end) = <class 'float'>; must be of type int 
Test # 136: Success, as expected.
Test # 137: Success, as expected.
Test # 138: Success, as expected.
Test # 139: Success, as expected.
Tests passed: 140. Tests failed: 0.

Testing ProfileMut() parameter err ...
Test # 140: Error, as expected:  type(err) = <class 'float'>; must be a boolean 
Test # 141: Error, as expected:  type(err) = <class 'str'>; must be a boolean 
Test # 142: Error, as expected:  type(err) = <class 'int'>; must be a boolean 
Test # 143: Error, as expected:  type(err) = <class 'str'>; must be a boolean 
Test # 144: Error, as expected:  type(err) = <class 'NoneType'>; must be a boolean 
Test # 145: Success, as expected.
Test # 146: Success, as expected.
Tests passed: 147. Tests failed: 0.

Error:  Not all sequences are the same length.
Testing ProfileInfo() parameter method ...
Test # 147: Error, as expected:  type(method) = <class 'float'>; must be a string 
Test # 148: Error, as expected:  method = x; must be in ['naive', 'tpm', 'nsb']
Test # 149: Error, as expected:  type(method) = <class 'int'>; must be a string 
Test # 150: Error, as expected:  method = True; must be in ['naive', 'tpm', 'nsb']
Test # 151: Error, as expected:  type(method) = <class 'NoneType'>; must be a string 
Test # 152: Success, as expected.
Test # 153: Success, as expected.
Tests passed: 154. Tests failed: 0.

Testing ProfileInfo() parameter pseudocount ...
Test # 154: Error, as expected:  type(pseudocount) = <class 'str'>; must be a float 
Test # 155: Error, as expected:  type(pseudocount) = <class 'int'>; must be a float 
Test # 156: Error, as expected:  type(pseudocount) = <class 'str'>; must be a float 
Test # 157: Error, as expected:  type(pseudocount) = <class 'NoneType'>; must be a float 
Test # 158: Error, as expected:  pseudocount is not nonnegative.
Test # 159: Success, as expected.
Test # 160: Success, as expected.
Test # 161: Success, as expected.
Tests passed: 162. Tests failed: 0.

Testing ProfileInfo() parameter start ...
Test # 162: Error, as expected:  type(start) = <class 'float'>; must be of type int 
Test # 163: Error, as expected:  type(start) = <class 'str'>; must be of type int 
Test # 164: Error, as expected:  type(start) = <class 'float'>; must be of type int 
Test # 165: Error, as expected:  type(start) = <class 'NoneType'>; must be of type int 
Test # 166: Success, as expected.
Test # 167: Success, as expected.
Test # 168: Success, as expected.
Test # 169: Success, as expected.
Tests passed: 170. Tests failed: 0.

Testing ProfileInfo() parameter end ...
Test # 170: Error, as expected:  type(end) = <class 'float'>; must be of type int 
Test # 171: Error, as expected:  type(end) = <class 'str'>; must be of type int 
Test # 172: Error, as expected:  type(end) = <class 'float'>; must be of type int 
Test # 173: Success, as expected.
Test # 174: Success, as expected.
Test # 175: Success, as expected.
Test # 176: Success, as expected.
Tests passed: 177. Tests failed: 0.

Testing LearnModel() parameter df ...
Test # 177: Error, as expected:   The Learn Model class requires pandas dataframe as input dataframe. Entered df was 'None'.
Test # 178: Error, as expected:  type(df) = <class 'float'>; must be a pandas dataframe 
Test # 179: Success, as expected.
Test # 180: Success, as expected.
Tests passed: 181. Tests failed: 0.

Testing LearnModel() parameter lm ...
Test # 181: Error, as expected:  type(lm) = <class 'NoneType'> must be a string 
Test # 182: Error, as expected:  type(lm) = <class 'float'> must be a string 
Test # 183: Error, as expected:  lm = x; must be in ['ER', 'LS', 'IM', 'PR']
Test # 184: Success, as expected.
Test # 185: Success, as expected.
Test # 186: Success, as expected.
Tests passed: 187. Tests failed: 0.

Testing EvaluateModel() parameter dataset_df ...
Test # 187: Error, as expected:   The Evaluate Model class requires pandas dataframe as input dataframe. Entered dataset_df was 'None'.
Test # 188: Error, as expected:  type(dataset_df) = <class 'float'>; must be a pandas dataframe 
Test # 189: Error, as expected:  type(dataset_df) = <class 'str'>; must be a pandas dataframe 
Test # 190: Success, as expected.
Tests passed: 191. Tests failed: 0.

Testing EvaluateModel() parameter model_df ...
Test # 191: Error, as expected:   The Evaluate Model class requires pandas dataframe as input model dataframe. Entered model_df was 'None'.
Test # 192: Error, as expected:  type(model_df) = <class 'float'>; must be a pandas dataframe 
Test # 193: Error, as expected:  type(model_df) = <class 'str'>; must be a pandas dataframe 
Test # 194: Success, as expected.
Tests passed: 195. Tests failed: 0.

Testing EvaluateModel() parameter left ...
Test # 195: Error, as expected:  type(left) = <class 'float'>; must be of type int 
Test # 196: Error, as expected:  type(left) = <class 'str'>; must be of type int 
Test # 197: Error, as expected:  Invalid start=-1
Test # 198: Error, as expected:  Invalid end=26 for seq_length=22
Test # 199: Success, as expected.
Test # 200: Success, as expected.
Tests passed: 201. Tests failed: 0.

Testing EvaluateModel() parameter right ...
Test # 201: Error, as expected:  type(right) = <class 'float'>; must be of type int 
Test # 202: Error, as expected:  type(right) = <class 'str'>; must be of type int 
Test # 203: Error, as expected:  Invalid start=-23
Test # 204: Error, as expected:  Invalid start=-18
Test # 205: Success, as expected.
Test # 206: Success, as expected.
Tests passed: 207. Tests failed: 0.
