"""
The simulate library class generates simulated data for a Sort Seq Experiment
with a given mutation rate and wild type sequence.
"""

import argparse
import numpy as np
import scipy as sp
import pandas as pd
import sys
import utils as utils
import qc as qc
import io_local as io
from mpathic import SortSeqError
from utils import check, handle_errors
import pdb
from numpy.random import choice

class SimulateLibrary:

    """

    Parameters
    ----------
    wtseq : (string)
            wildtype sequence. Must contain characteres 'A', 'C', 'G','T' for \n
            dicttype = 'DNA', 'A', 'C', 'G','U' for  dicttype = 'RNA'

    mutrate : (float)
        mutation rate

    numseq : (int)
        number of sequences. Must be a positive integer.

    dicttype : (string)
        sequence dictionary: valid choices include 'dna', 'rna', 'pro'

    probarr : (np.ndarray)
        probability matrix used to generate bases

    tags : (boolean)
        If simulating tags, each generated seq gets a unique tag

    tag_length : (int)
        Length of tags. Should be >= 0


    Attributes
    ----------
    output_df : (pandas dataframe)
        Contains the output of simulate library in a pandas dataframe.


    """

    # main function for simulating library
    @handle_errors
    def __init__(
                 self,
                 wtseq = "ACGACGA",
                 mutrate=0.10,
                 numseq=10000,
                 dicttype='dna',
                 probarr=None,
                 tags=False,
                 tag_length=10):

        # setting attributes to parameters. This could be modified.
        self.wtseq = wtseq
        self.mutrate = mutrate
        self.numseq = numseq
        self.dicttype = dicttype
        self.probarr = probarr
        self.tags = tags
        self.tag_length = tag_length
        # attribute that gets populated after running the constructor
        self.output_df = None

        # Validate inputs:
        self._input_check()

        # generate sequence dictionary
        seq_dict, inv_dict = utils.choose_dict(dicttype)

        if isinstance(probarr,np.ndarray):
            L = probarr.shape[1]
            #Generate bases according to provided probability matrix
            letarr = np.zeros([numseq,L])
            for z in range(L):
                letarr[:,z] = np.random.choice(
                    range(len(seq_dict)),numseq,p=probarr[:,z])
        else:
            parr = []
            wtseq = wtseq.upper()
            L = len(wtseq)
            letarr = np.zeros([numseq,L])

            #find wtseq array
            wtarr = self.seq2arr(wtseq,seq_dict)
            mrate = mutrate/(len(seq_dict)-1)  # prob of non wildtype
            # Generate sequences by mutating away from wildtype
            '''probabilities away from wildtype (0 = stays the same, a 3 for 
                example means a C becomes an A, a 1 means C-> G)'''
            parr = np.array(
                [1-(len(seq_dict)-1)*mrate]
                + [mrate for i in range(len(seq_dict)-1)])
            # Generate random movements from wtseq
            letarr = np.random.choice(
                range(len(seq_dict)),[numseq,len(wtseq)],p=parr)
            #Find sequences
            letarr = np.mod(letarr + wtarr,len(seq_dict))
        seqs= []
        # Convert Back to letters
        for i in range(numseq):
            seqs.append(self.arr2seq(letarr[i,:],inv_dict))

        seq_col = qc.seqtype_to_seqcolname_dict[dicttype]
        seqs_df = pd.DataFrame(seqs, columns=[seq_col])

        # If simulating tags, each generated seq gets a unique tag
        if tags:
            tag_seq_dict,tag_inv_dict = utils.choose_dict('dna')
            tag_alphabet_list = tag_seq_dict.keys()

            check(len(tag_alphabet_list) ** tag_length > 2 * numseq,
                  'tag_length=%d is too short for num_tags_needed=%d' % (tag_length, numseq))

            # Generate a unique tag for each unique sequence
            tag_set = set([])
            while len(tag_set) < numseq:
                num_tags_left = numseq - len(tag_set)
                new_tags = [''.join(choice(tag_alphabet_list,size=tag_length)) \
                    for i in range(num_tags_left)]
                tag_set = tag_set.union(new_tags)

            df = seqs_df.copy()
            df.loc[:,'ct'] = 1
            df.loc[:,'tag'] = list(tag_set)

        # If not simulating tags, list only unique seqs w/ corresponding counts
        else:
            seqs_counts = seqs_df[seq_col].value_counts()
            df = seqs_counts.reset_index()
            df.columns = [seq_col,'ct']

        # Convert into valid dataset dataframe and return
        self.output_df = qc.validate_dataset(df,fix=True)
        #print(self.output_df.head())


    def seq2arr(self,seq,seq_dict):
        """
        Change base pairs to numbers
        """
        return np.array([seq_dict[let] for let in seq])

    def arr2seq(self,arr,inv_dict):
        """
        Change numbers back into base pairs.
        """
        return ''.join([inv_dict[num] for num in arr])

    def _input_check(self):
        """
        Check all parameter values for correctness

        """

        ########################
        #  wtseq input checks  #
        ########################

        # check if wtseq is of type string
        check(isinstance(self.wtseq,str),'type(wtseq) = %s; must be a string ' % type(self.wtseq))

        # check if empty wtseq is passed
        check(len(self.wtseq) > 0, "wtseq length cannot be 0")

        # Check to ensure the wtseq uses the correct bases according to dicttype

        # unique characters in the wtseq parameter as a list
        unique_base_list = list(set(self.wtseq))

        # if more than 4 unique bases detected and dicttype is not protein
        if(len(unique_base_list)>4 and self.dicttype!='protein'):
            print(' Warning, more than 4 unique bases detected for dicttype %s did you mean to enter protein for dicttype? ' % self.dicttype)

        # if 'U' base detected and dicttype is not 'rna'
        if('U' in unique_base_list and self.dicttype!='rna'):
            print(' Warning, U bases detected for dicttype %s did you mean to enter rna for dicttype? ' % self.dicttype)

        lin_seq_dict, lin_inv_dict = utils.choose_dict(self.dicttype, modeltype='MAT')
        check(set(self.wtseq).issubset(lin_seq_dict),'wtseq can only contain bases in ' + str(lin_seq_dict.keys()))

        ##########################
        #  mutrate input checks  #
        ##########################

        # check if mutrate is of type float
        check(isinstance(self.mutrate, float), 'type(mutrate) = %s; must be a float ' % type(self.mutrate))

        # ensure mutrate is in the correct range
        check(self.mutrate > 0 and self.mutrate <= 1,'mutrate = %d; must be %d <= mutrate <= %d.' %
              (self.mutrate, 0, 1))

        #########################
        #  numseq input checks  #
        #########################

        # check if numseq is valid
        check(isinstance(self.numseq, int), 'type(numseq) = %s; must be a int ' % type(self.numseq))

        # check if numseq is positive
        check(self.numseq > 0, 'numseq = %d must be a positive int ' % self.numseq)

        ###########################
        #  dicttype input checks  #
        ###########################

        # check if dicttype is of type string
        check(isinstance(self.dicttype, str), 'type(dicttype) = %s; must be a string ' % type(self.dicttype))

        # check if len(dicttype) > 0
        check(len(self.dicttype) > 0,
              " length of dicttype must be greater than 0, length(dicttype): %d" % len(self.dicttype))

        ###########################
        #  probarr input checks   #
        ###########################

        # check if probarr is an ndarray
        if self.probarr is not None:
            check(isinstance(self.probarr, np.ndarray), 'type(probarr) = %s; must be an np.ndarray ' % type(self.probarr))

        #######################
        #  tags input checks  #
        #######################

        # *** NOTE ***: an additional check is made on tags in the constructor if tags = True

        # check if tags is of type bool.
        check(isinstance(self.tags, bool), 'type(tags) = %s; must be an boolean ' % type(self.tags))

        #############################
        #  tag_length input checks  #
        #############################

        # check if tag_length is of type int
        check(isinstance(self.tag_length, int), 'type(tag_length) = %s; must be an int ' % type(self.tag_length))

        # check if tag_length is of positive
        check(self.tag_length > 0, 'tag_length = %d must be a positive int ' % self.tag_length)


# /usr/local/Cellar/python3/3.6.2/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/