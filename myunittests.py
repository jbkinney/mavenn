#!/usr/bin/env python
import csv
import numpy as np
import os
import unittest
import sys
from Bio import SeqIO
from pandas.util.testing import assert_frame_equal
import sst.utils as utils
import sst.preprocess as preprocess
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna
import sst.learn_matrix as learn_matrix
import sst.predictiveinfo as predictiveinfo
import numpy.testing as npt

class TestPreprocessingCompletion(unittest.TestCase):
	'''Create separate Fasta files, then test that A) the script will run
            correctly given proper inputs. B) the script will throw a type error
            if any of the files are not fasta or fastq C) the script will throw a
            type error if the fileslist data frame input does not have the correct
            columns '''
	def setUp(self):
                '''Construct input list (bin,filename) and the files themselves)'''
		self.test_df_list = pd.DataFrame([0,1,2])
                self.test_df_list.columns = ['bin']
                self.test_df_list['file'] = ['test_fasta_1.fasta','test_fasta_2.fasta','test_fasta_3.fasta']
                #construct list with incorrect file names (no .fasta or .fastq)
                self.test_df_list_err = pd.DataFrame([0,1,2])
                self.test_df_list_err.columns = ['bin']
                self.test_df_list_err['file'] = ['test_fasta_1.fasta','test_fasta_2.fasta','test_fasta_3']
		pd.set_option('max_colwidth',int(1e8)) # make sure columns are not truncated
                self.test_df_list.to_string(
                    open('test_filelist','w'), index=False,col_space=10,float_format=utils.format_string)
                #construct the files
                seqs1 = ['ACG','ACT','ACG']
                seqs2 = ['ACC','ACG','ACT']
                seqs3 = ['ACG','ATT','ACT']
                #create seq record objects from seq list
                seqs1 = [SeqRecord(Seq(seqs1[i],generic_dna)) for i in range(len(seqs1))]
                seqs2 = [SeqRecord(Seq(seqs2[i],generic_dna)) for i in range(len(seqs2))]
                seqs3 = [SeqRecord(Seq(seqs3[i],generic_dna)) for i in range(len(seqs3))]
                SeqIO.write(seqs1,open('test_fasta_1.fasta','w'),'fasta')
                SeqIO.write(seqs2,open('test_fasta_2.fasta','w'),'fasta')
                SeqIO.write(seqs3,open('test_fasta_3.fasta','w'),'fasta')

	def test_completion(self):
		'''test that we get the correct results when we run'''
	        correct_dataframe = pd.DataFrame([['ACG',2,1,1],['ACT',1,1,1],['ACC',0,1,0],['ATT',0,0,1]],columns=['seq','ct_0','ct_1','ct_2'])
                #run program
                os.system('''sst preprocess -i test_filelist -o test_preprocessing_df''')
                #read in results
                df_test = pd.io.parsers.read_csv('test_preprocessing_df',delim_whitespace=True)
                #do conversions such that we can compare properly
                #convert to int
                df_test[['ct_0','ct_1','ct_2']] = df_test[['ct_0','ct_1','ct_2']].astype(int)
                #sort the dataframes by sequence and make sure the indexes are reset
                df_test.sort(columns='seq',inplace=True)
                df_test.reset_index(inplace=True,drop=True)
                correct_dataframe.sort(columns='seq',inplace=True)
                correct_dataframe.reset_index(inplace=True,drop=True)
                #check that the frames match
                assert_frame_equal(correct_dataframe,df_test)
                os.system('''rm test_preprocessing_df''')
		
	def test_fasta_fastq_error(self):
                self.assertRaises(
                    TypeError,lambda: preprocess.main(self.test_df_list_err))
        def test_correct_columns_error(self):
                columns_test_df_list = self.test_df_list
                columns_test_df_list.rename(columns={'bin':'incorrect'},inplace=True)
                self.assertRaises(
                    TypeError,lambda: preprocess.main(columns_test_df_list))
	def tearDown(self):
		'''Delete All created files'''
                os.system('''rm test_fasta_*.fasta''')
                os.system('''rm test_filelist''')

class TestLearnMatrix(unittest.TestCase):
    '''We will test that the program can recapitulate the model we previously
        calculated on the test data (least squares only). We will also make sure
        that it throws an error when confronted with inputs that do are not
        of the correct type (dna,rna,protein).'''

    def setUp(self):
        #load in crp-wt data file
               
        self.data_df = pd.io.parsers.read_csv(
                           os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'data/sortseq/crp-wt/data.txt'),
                           delim_whitespace=True)
        #load in crp model for comparison
        self.crp_model = pd.io.parsers.read_csv(
                           os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'data/sortseq/crp-wt/test_old'),
                           delim_whitespace=True)
    def test_least_squares(self):
        #learn a model for crp-wt
        df = learn_matrix.main(self.data_df,'dna','leastsq',start=3,end=25)
        npt.assert_allclose(df,self.crp_model,atol=.001)

    def test_catches_incorrect_alphabet(self):
        data_df_test = self.data_df
        #change one record to have improper entries
        data_df_test.loc[0,'seq'] = ''.join(
                                 ['F' for i in range(len(data_df_test.loc[0,'seq']))])
        self.assertRaises(
            TypeError,lambda: learn_matrix.main(data_df_test,'dna','leastsq'))

    def tearDown(self):
        pass

class TestPredictiveInformation(unittest.TestCase):
    def setUp(self):
        #load in crp-wt data file
               
        self.data_df = pd.io.parsers.read_csv(
                           os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'data/sortseq/crp-wt/data.txt'),
                           delim_whitespace=True)
        #load in crp model for comparison
        self.crp_model = pd.io.parsers.read_csv(
                           os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'data/sortseq/crp-wt/test_old'),
                           delim_whitespace=True)
        #the previously found predictive information
        self.pred_info = 0.532794

    def test_predictive_info_on_crp(self):
        MI,Std = predictiveinfo.main(self.data_df,self.crp_model,start=3,end=25)
        self.assertAlmostEqual(MI,self.pred_info,places=4)

    def tearDown(self):
        pass
if __name__ == '__main__':
    unittest.main()
		
			
