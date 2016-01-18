#!/usr/bin/env python

'''A scripts which accepts a dataframe of input files for each bin and outputs
    a single data frame containing all the info. Combining of paired end reads,
    and any quality score filtering must occur before this script runs.'''
from __future__ import division
#Our standard Modules
import argparse
import numpy as np
import sys
import pandas as pd
import sst.utils as utils
from Bio import SeqIO
import sst.io

import qc

fasta_filename_patterns=r'(.fasta$|.fas$|.fsa$|.ffn$|.fna$|.fa|.frn|.faa$)'
fastq_filename_patterns=r'(.fastq$|.fq$)'

# def main(filelist_df,tags_df=None):

#     # Validate filelist
#     qc.validate(filelist_df)

#     # Read datasets into dictionary indexed by bin number
#     dataset_df_dict = {}
#     for item in filelist_df.iterrows():
#         # Autodetect fasta, fastq, or text file based on file extension
#         fn = item[1]['file']
#         b = item[1]['bin']
#         if re.search(fasta_filename_patterns,fn:
#             df = load_dataset(fn,file_type='fasta')
#         elif re.search(fastq_filename_patterns,fn):
#             df = load_dataset(fn,file_type='fastq')
#         else:
#             df = load_dataset(fn,file_type='text')
#         dataset_df_dict[b] = df

#     # Make sure datasets were loaded
#     if not len(dataset_df_dict)>=1:
#         raise TypeError('No datasets were loaded')

#     # Determine index column. Must be same for all files
#     df = dataset_df_dict.values()[0]
#     if 'tag' in df.columns:
#         index_col = 'tag'
#     elif 'seq' in df.columns:
#         index_col = 'seq'
#     elif 'seq_rna' in df.columns:
#         index_col = 'seq_rna'
#     elif 'seq_pro' in df.columns:
#         index_col = 'seq_pro'

#     # Concatenate dataset dataframes
#     for b in dataset_df_dict.keys()
#         df = dataset_df_dict[b]

#         # Verify that dataframe has correct column
#         if not index_col in df.colums:
#             raise TypeError('\
#                 Dataframe does not contain index_col="%s"'%index_col)
#         if not 'ct' in df.columns:
#             raise TypeError('\
#                 Dataframe does not contain a "ct" column')

#         # Delete "ct_X" columns
#         df.drop([c for c in df.columns if is_col_type(col_name,'ct_')])

#         # Index dataset by index_col
#         df.groupby(index_col)

#     # Concatenate datasets


def main(filelist_df,tags_df=None):

    # Validate input dataframes
    qc.validate_filelist(filelist_df)
    if not tags_df is None:
        qc.validate_tagkey(tags_df)
    
    for item in filelist_df.iterrows():
        '''If files are fasta or fastq, convert them to dataframe, otherwise throw
            error'''
        fn = item[1]['file']
        temp_df = pd.DataFrame()
        if '.fasta' in fn:
            df = pd.DataFrame(columns=['seq'])
            #Read each sequence record into one row of dataframe
            for i,record in enumerate(SeqIO.parse(fn,'fasta')):
                 df.loc[i] = str(record.seq)
        elif '.fastq' in fn:
            df = pd.DataFrame(columns=['seq'])
            for i,record in enumerate(SeqIO.parse(fn,'fastq')):
                 df.loc[i] = str(record.seq)
        else:
            raise TypeError('input File must be Fasta or FastQ')
        #If no sum of counts column exists, assign each seq a count of 1
        try:
            ct = df['ct']
        except:
            df['ct'] = 1
        #if the experiment is an mpra experiment, translate tags
        if isinstance(tags_df,pd.DataFrame):
            #if there are tags, count the number of each tag
            df = df.groupby('tag').sum()                
        else:
            #count number of each unique sequence.      
            df = df.groupby('seq').sum()
        #now add this file to a total output dataframe
        temp_df['ct_' + str(item[1]['bin'])] = df['ct']
        output_df = pd.concat([output_df,temp_df],axis=1)
    '''currently, the sequence/tag will be set as the index instead of a column.
         if we reset index it will become a column. The name of this column will
         be 'index', which will will need to rename in a few lines.'''
    output_df = output_df.reset_index()
    #if the experiment is mpra, connect each tag with corresponding sequence
    if isinstance(tags_df,pd.DataFrame):
        output_df = output_df.rename(columns = {'index':'tag'})       
        output_df['seq'] = [
            tags_df['seq'][item[1]['tag']] for item in output_df.iterrows()] 
    else:
        output_df = output_df.rename(columns = {'index':'seq'})
    '''any bin without where a sequence doesnt apprear will report an NA, we 
        will now change this to 0'''
    output_df = output_df.fillna(value=0)
    
    return output_df

# Define commandline wrapper
def wrapper(args):  
    
    # Run funciton
    if args.i:
        #filelist_df = pd.io.parsers.read_csv(args.i,delim_whitespace=True)
        filelist_df = sst.io.load_filelist(args.i)
    else:
        #filelist_df = pd.io.parsers.read_csv(sys.stdin,delim_whitespace=True)
        filelist_df = sst.io.load_filelist(sys.stdin)
    
    if args.tagkeys:
        #tags_df = pd.io.parsers.read_csv(args.tagkeys,delim_whitespace=True)
        tags_df = sst.io.load_tagkeys(args.tagkeys)
    else:
        tags_df = None

    output_df = main(filelist_df,tags_df=tags_df)
    
    

    if args.out:
        outloc = open(args.out,'w')
    else:
        outloc = sys.stdout
    pd.set_option('max_colwidth',int(1e8)) # make sure columns are not truncated
    output_df.to_string(
        outloc, index=False,col_space=10,float_format=utils.format_string)

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('preprocess')
    p.add_argument(
        '-i', '--i', default=None,help='''Input file, otherwise input
        through the standard input.''')
    p.add_argument('--tagkeys',default=None)
    p.add_argument('-o', '--out', default=None)
    p.set_defaults(func=wrapper)
