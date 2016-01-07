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



def main(filelist_df,tags_df=None):

    #make sure input columns are bin and file
    columns = set(filelist_df.columns)
    if not columns == {'bin','file'}:
        raise TypeError('Incorrect column headers for fileslist dataframe!')

    output_df = pd.DataFrame()
    #if there are tags it is an mpra exp. We need to change a few things...
    try:
        tags_df = tags_df.set_index('tag')
    except:
        pass
    
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
        filelist_df = pd.io.parsers.read_csv(args.i,delim_whitespace=True)
    else:
        filelist_df = pd.io.parsers.read_csv(sys.stdin,delim_whitespace=True)
    
    if args.tagkeys:
        tags_df = pd.io.parsers.read_csv(args.tagkeys,delim_whitespace=True)
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
