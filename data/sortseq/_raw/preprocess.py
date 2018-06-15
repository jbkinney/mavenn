#!/usr/bin/env python2.7
import os, sys
from collections import Counter
from sets import Set
import shutil
import gzip

# Specify input files
reads_file = 'reads.fa.gz'
primers_file = 'primers.txt'
bin_codes_file = 'bin_codes.txt'
out_dir = '..'

# Load bin codes
g = open(bin_codes_file )
glines = g.readlines()[1:]
g.close()
code_to_filename_dict = {}
experiments = Set([])
for l in glines:
	atoms = l.split()
	code = int(atoms[0])
	experiment = atoms[1]
	thebin = atoms[2]
	filename = '%s/%s/%s.txt'%(out_dir,experiment,thebin)
	code_to_filename_dict[code] = filename
	experiments.add(out_dir + '/' + experiment)

# Load primer sequences
print('Loading primer sequences...')
g = open(primers_file)
glines = g.readlines()[0:47]
g.close()
barcodetable = {}
for l in glines:
	atoms = l.split()
	z = int(atoms[0][-2:])
	barcode = atoms[1][19:26]
	barcodetable[barcode] = z

# Extract sequences from raw data file
print('Loading raw data file...')
f = gzip.open(reads_file, 'r')
lines = f.readlines()
f.close()
seqs = []
s = ''
print('Extracting reads from data file...')
k=0;
for l in lines:
	if '>' in l:
		k=k+1;
		#if k%1000 == 0:
			#print('On sequence ' + str(k))
		if len(s) > 1:
			seqs.append(s)
		s = ''
	else:
		s = s + l.strip()
		
# Filter sequences
data = [] 
print('Filtering seqs...')
for l in seqs:
	b = l.find('AATTGTGAGC');
	a = l.find('AGCGCAACGC') + 10;
	barcode = l[0:7];
	if (b-a == 75) and (barcode in barcodetable.keys()):
		z = barcodetable[barcode]
		seq = l[a:b]
		if not ('N' in l[a:b]):
			data.append([z, seq])
			if not len(seq) == 75:
				print('LENGTH ERROR!')

# Clean directories for output
print('Preparing directories for output...')
for experiment in experiments:
	if os.path.exists(experiment):
		shutil.rmtree(experiment)
	os.mkdir(experiment)

# Write sequences for each experiment
print('Writing sequences of reach experiment...')
for code in code_to_filename_dict.keys():

	# Get sequences for specified code
	seqs = [a[1] for a in data if a[0]==code]

	# Count the number of occurances of each sequence
	counter = Counter(seqs)

	# Get unique sequences and corresponding counts
	seq_ct_list = counter.most_common()

	# Write sequences and counts to file
	filename = code_to_filename_dict[code]
	print('Writing %s ...'%filename)
	h = open(filename, 'w')
	h.write('ct\tseq\n')
	for seq, ct in seq_ct_list:
		h.write('%s\t%s\n'%(ct,seq))
	h.close()
	
print('Done!')


