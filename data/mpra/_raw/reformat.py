#!/usr/bin/env python2.7
import os, glob
from sets import Set
import gzip, shutil

os.system('rm -r ../*Rep*')

outdir = '..'
infile_names = glob.glob('*Multi*.counts.txt.gz')

# Parse out experiment names
experiments = Set(['_'.join(name.split('_')[1:5]) for name in infile_names])

# For each experiment
for experiment in experiments:
	print('---------\nProcessing experiment %s...'%experiment)

	# Create clean output directory 
	experiment_dir = '%s/%s'%(outdir,experiment)
	if os.path.exists(experiment_dir):
		shutil.rmtree(experiment_dir)
	os.mkdir(experiment_dir)

	# Set library and expression files
	library_filename = [x for x in infile_names 
		if (experiment in x) and ('Plasmid' in x)][0]
	expression_filename = [x for x in infile_names 
		if (experiment in x) and ('mRNA' in x)][0]

	# Create tag key
	alltags = Set()
	tagkey_filename = '%s/key.txt'%experiment_dir
	print('Writing %s...'%tagkey_filename)
	outfile = open(tagkey_filename,'w')
	outfile.write('tag\tseq\n')
	inlines = gzip.open(library_filename, 'r').readlines()[1:]
	for l in inlines:
		atoms = l.split()
		seq = atoms[1]
		# This is necessary becomes sometimes lines have multiple tags
		tags = atoms[2].split(',')
		for tag in tags:
			outfile.write('%s\t%s\n'%(tag,seq))
			alltags.add(tag)  # Add to list of tags in key
	outfile.close()

	# Create library file
	newlibrary_filename = '%s/library.txt'%experiment_dir
	print('Writing %s...'%newlibrary_filename)
	outfile = open(newlibrary_filename,'w')
	outfile.write('ct\ttag\n')
	inlines = gzip.open(library_filename, 'r').readlines()[1:]
	for l in inlines:
		atoms = l.split()
		tags = atoms[2].split(',')
		cts = atoms[3].split(',')
		for ct, tag in zip(cts,tags):
			assert tag in alltags  # Make sure tag is in key
			outfile.write('%5d\t%s\n'%(int(ct),tag))
	outfile.close()

	# Create expression file
	newexpression_filename = '%s/expression.txt'%experiment_dir
	print('Writing %s...'%newexpression_filename)
	outfile = open(newexpression_filename,'w')
	outfile.write('ct\ttag\n')
	inlines = gzip.open(expression_filename, 'r').readlines()[1:]
	for l in inlines:
		atoms = l.split()
		tags = atoms[2].split(',')
		cts = atoms[3].split(',')
		for ct, tag in zip(cts,tags):
			assert tag in alltags  # Make sure tag is in key
			outfile.write('%5d\t%s\n'%(int(ct),tag))
	outfile.close()

	files_filename = '%s/files.txt'%experiment_dir
	print('Writing %s...'%files_filename)
	outfile = open(files_filename,'w')
	s = '''bin  file
  0  library.txt
  1  expression.txt
'''
	outfile.write(s)
	outfile.close()

print('Done!')
