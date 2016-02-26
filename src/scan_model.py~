#!/usr/bin/env python

'''A script which accepts an model of binding energy and a wild type sequence.
    The script scans the model across the sequence, and generates an energy
    prediction for each starting position. It then sorts by best binding and 
    displays all posibilities.'''
from __future__ import division
#Our standard Modules
import argparse
import numpy as np
import scipy as sp
import sys
#Our miscellaneous functions
import pandas as pd
from Bio import SeqIO
import sortseq_tools.utils as utils
import sortseq_tools.Models as Models
import sortseq_tools.io as io
import sortseq_tools.qc as qc
import sortseq_tools.fast as fast
import re
import pdb
from sortseq_tools import SortSeqError

def main(model_df, contig_list, numsites=10, verbose=False):

    # Determine type of string from model
    qc.validate_model(model_df)
    seqtype, modeltype = qc.get_model_type(model_df)
    seq_dict,inv_dict = utils.choose_dict(seqtype,modeltype=modeltype)

    # Check that all characters are from the correct alphabet
    alphabet = qc.seqtype_to_alphabet_dict[seqtype]
    search_string = r"[^%s]"%alphabet
    for contig_str, contig_name, pos_offset in contig_list:
        if re.search(search_string,contig_str):
            raise SortSeqError(\
                'Invalid character for seqtype %s found in %s.'%\
                (seqtype,contig_name))

    # Create model object to evaluate on seqs
    if modeltype == 'MAT':
        model_obj = Models.LinearModel(model_df)
    elif modeltype == 'NBR':
        model_obj = Models.NeighborModel(model_df)
    
    # Create list of dataframes, one for each contig
    seq_col = qc.seqtype_to_seqcolname_dict[seqtype]
    L = model_obj.length
    sitelist_df = pd.DataFrame(\
            columns=['val',seq_col,'left','right','ori','contig'])
    for contig_str, contig_name, pos_offset in contig_list:
        if len(contig_str) < L:
            continue
        this_df = pd.DataFrame(\
            columns=['val',seq_col,'left','right','ori','contig'])
        num_sites = len(contig_str) - L + 1
        poss = np.arange(num_sites).astype(int) 
        this_df['left'] = poss + pos_offset
        this_df['right']  = poss + pos_offset + L - 1 
        #this_df[seq_col] = [contig_str[i:(i+L)] for i in poss]
        this_df[seq_col] = fast.seq2sitelist(contig_str,L)  #Cython
        this_df['ori'] = '+'
        this_df['contig'] = contig_name
        this_df['val'] = model_obj.evaluate(this_df[seq_col])
        sitelist_df = pd.concat([sitelist_df,this_df], ignore_index=True)

        # If scanning DNA, scan reverse-complement as well
        if seqtype=='dna':
            #this_df[seq_col] = [qc.rc(s) for s in this_df[seq_col]]
            this_df[seq_col] = fast.seq2sitelist(contig_str,L,rc=True)  #Cython
            this_df['ori'] = '-'
            this_df['val'] = model_obj.evaluate(this_df[seq_col])
            sitelist_df = pd.concat([sitelist_df,this_df], ignore_index=True)

        # Sort by value and reindex
        sitelist_df.sort_values(by='val', ascending=False, inplace=True)
        sitelist_df.reset_index(drop=True,inplace=True)

        # Crop list at numsites
        if sitelist_df.shape[0]>numsites:
            sitelist_df.drop(sitelist_df.index[numsites:], inplace=True)

        if verbose:
            print '.',
            sys.stdout.flush()

    if verbose:
        print ''
        sys.stdout.flush()

    # If no sites were found, raise error
    if sitelist_df.shape[0]==0:
        raise SortSeqError(\
            'No full-length sites found within provided contigs.')

    sitelist_df = qc.validate_sitelist(sitelist_df,fix=True)
    return sitelist_df


def wrapper(args):
    """ Wrapper for function for scan_model.main()
    """

    # Prepare input to main
    model_df = io.load_model(args.model)
    seqtype, modeltype = qc.get_model_type(model_df)
    L = model_df.shape[0]
    if modeltype=='NBR':
        L += 1 
    
    chunksize = args.chunksize
    if not chunksize>0:
        raise SortSeqError(\
            'chunksize=%d must be positive'%chunksize)

    if args.numsites <= 0:
        raise SortSeqError('numsites=%d must be positive.'%args.numsites)

    if args.i and args.seq:
        raise SortSeqError('Cannot use flags -i and -s simultaneously.')

    # If sequence is provided manually
    if args.seq:
        pos_offset=0
        contig_str = args.seq

        # Add a bit on end if circular
        if args.circular:
            contig_str += contig_str[:L-1] 

        contig_list = [(contig_str,'manual',pos_offset)]

    # Otherwise, read sequence from FASTA file
    else:
        contig_list = []
        inloc = io.validate_file_for_reading(args.i) if args.i else sys.stdin
        for i,record in enumerate(SeqIO.parse(inloc,'fasta')):
            name = record.name if record.name else 'contig_%d'%i

             # Split contig up into chunk)size bits
            full_contig_str = str(record.seq)

            # Add a bit on end if circular
            if args.circular:
                full_contig_str += full_contig_str[:L-1] 

            # Define chunks containing chunksize sites
            start = 0
            end = start+chunksize+L-1
            while end < len(full_contig_str):
                contig_str = full_contig_str[start:end]
                contig_list.append((contig_str,name,start))
                start += chunksize
                end = start+chunksize+L-1
            contig_str = full_contig_str[start:]
            contig_list.append((contig_str,name,start))

        if len(contig_list)==0:
            raise SortSeqError('No input sequences to read.')

    # Compute results
    outloc = io.validate_file_for_writing(args.out) if args.out else sys.stdout
    output_df = main(model_df,contig_list,numsites=args.numsites,\
        verbose=args.verbose)

    # Write df to stdout or to outfile 
    io.write(output_df,outloc,fast=args.fast)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('scan_model')
    p.add_argument(
        '-s','--seq',type=str,default=None,
        help ='manually enter sequence to scan')
    p.add_argument(
        '-i','--i',type=str,default=None,
        help ='specify FASTA sequence contig file')
    p.add_argument(
        '-m','--model', help='model to scan sequence with')
    p.add_argument('-o', '--out', default=None)
    p.add_argument(
        '-f','--fast', action='store_true', 
        help="Output is a little harder to read, but is written much faster."
        )
    p.add_argument(
        '-n','--numsites', type=int, default=10, 
        help="Maximum number of sites to record. Positive integer."
        )
    p.add_argument(
        '-k','--chunksize', type=int, default=100000, 
        help="chunksize to use when parsing FASTA files. Positive integer."
        )
    p.add_argument(
        '-c','--circular', action='store_true', 
        help="Treat sequences as circular."
        )
    p.add_argument(
        '-v','--verbose', action='store_true', 
        help="provides updates on scan. "
        )
    p.set_defaults(func=wrapper)
