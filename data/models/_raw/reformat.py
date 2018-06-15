#!/usr/bin/env python2.7
import os, glob

outdir = '..'
oldnames = glob.glob('*.matrix')
for filename in oldnames:
	newfilename = '%s/%s.txt'%(outdir,'.'.join(filename.split('.')[:-1]))

	print('Reformatting %s -> %s ...'%(filename,newfilename))

	f = open(filename,'r')
	lines = [l for l in f.readlines() if len(l.split())==4]
	g = open(newfilename,'w')
	g.write('pos\tpar_A\tpar_C\tpar_G\tpar_T\n')
	for i, l in enumerate(lines):
		float_format = '%0.2f\t'
		atoms = l.split()
		line = '%d\t'%i
		for a in atoms:
			line += float_format%float(a)
		line += '\n'
		g.write(line)
