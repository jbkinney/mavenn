import numpy as np

# Convert a sequence of length L of ACGT to a number
dna_base_to_index_dict = {'A':0,'C':1,'G':2,'T':3}

def dna2mat(seq):
	L = len(seq)
	arr = np.zeros(4*L).astype(int)
	for i, c in enumerate(seq):
		b = dna_base_to_index_dict[c]
		arr[4*i+b] = 1
	return arr

if __name__=='__main__':
	print dna2mat('ACACGTA')