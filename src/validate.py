from __future__ import division
import argparse
import sys
import io_local as io
import qc as qc
import re
import pdb
#from . import SortSeqError
from mpathic import SortSeqError

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
    'tagkey'        :   io.load_tagkey,
    'meanstd'       :   io.load_meanstd,
    'sitelist'      :   io.load_sitelist
}
filetypes = filetype_to_loadfunc_dict.keys()

# Define commandline wrapper
def wrapper(args):
    """ Wrapper for functions io.load_* and io.write
    """  

    # Determine input and output
    inloc = io.validate_file_for_reading(args.i) if args.i else sys.stdin
    outloc = io.validate_file_for_writing(args.out) if args.out else sys.stdout

    try:
        # Get load function corresponding to file type
        func = filetype_to_loadfunc_dict[str(args.type)]

        # Run load function on input
        df = func(inloc)

        # Write df to stdout or to outfile 
        io.write(df,outloc,fast=args.fast)

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
        '-t', '--type', required=True, choices=filetypes, \
        help="Type of file to validate input as.")
    p.add_argument(
        '-f','--fast', action='store_true', 
        help="Output is a little harder to read, but is written much faster."
        )
    p.set_defaults(func=wrapper)
