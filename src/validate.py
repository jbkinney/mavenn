from __future__ import division
import argparse
import sys
import sst.io as io
import sst.qc as qc
import re
import pdb
from sst import SortSeqError

# Filetypes and corrsponding load functions
filetype_to_loadfunc_dict = {
    'filelist'      :   io.load_filelist,
    'profile_info'  :   io.load_profile_info,
    'profile_mut'   :   io.load_profile_mut,
    'profile_ct'    :   io.load_profile_ct,
    'profile_freq'  :   io.load_profile_freq,
    'dataset'       :   io.load_dataset,
    'dataset_fasta_dna'     :   \
        lambda f: io.load_dataset(f,file_type='fasta',seq_type='dna'),
    'dataset_fasta_rna'     :   \
        lambda f: io.load_dataset(f,file_type='fasta',seq_type='rna'),
    'dataset_fasta_protein' :   \
        lambda f: io.load_dataset(f,file_type='fasta',seq_type='protein'),
    'dataset_fastq' :   \
        lambda f: io.load_dataset(f,file_type='fastq'),
    'model'         :   io.load_model,
    'tagkey'        :   io.load_tagkey
}
filetypes = filetype_to_loadfunc_dict.keys()

# Define commandline wrapper
def wrapper(args):
    """ Wrapper for functions io.load_* and io.write
    """  
    
    # Determine input and output
    inloc = open(args.i,'r') if args.i else sys.stdin
    outloc = open(args.out,'w') if args.out else sys.stdout

    try:
        # Get load function corresponding to file type
        func = filetype_to_loadfunc_dict[str(args.filetype)]

        # Run load function on input
        df = func(inloc)

        # Write df to stdout or to outfile 
        io.write(df,outloc)

    except SortSeqError:
        raise
    #    raise SortSeqError('Could not interpret input as %s'%args.filetype)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('validate')
    p.add_argument(
        '-i', '--i', default=None,help='''Input file, otherwise input
        through the standard input.''')
    p.add_argument('-o', '--out', default=None)
    p.add_argument(
        '-f', '--filetype', required=True, choices=filetypes, \
        help='''Type of sequence to expect in input files.''')
    p.set_defaults(func=wrapper)