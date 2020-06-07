#!/usr/bin/env python

in_files = '''
run0cor.txt.out
run3cor.txt.out
run6cor.txt.out
'''.split()

out_files = '''
library.txt
bin_1.txt
bin_2.txt
'''.split()

out_dir = '../'

for n in range(3):
	f = open(in_files[n],'r')
	g = open(out_dir + out_files[n],'w')
	
	# Read in data, organize, and sort
	atoms_list = [line.split() for line in f.readlines()]
	tuples_list = [(int(atoms[1]),atoms[0]) for atoms in atoms_list 
		if len(atoms)==2]
	sorted_tuples_list = sorted(tuples_list, reverse=True, key=lambda x: x[0])

	# Write to output file
	g.write('ct\tseq_pro\n')
	for tup in sorted_tuples_list:
		g.write('%8d\t%s\n'%tup)

	# Close files
	f.close()
	g.close()
