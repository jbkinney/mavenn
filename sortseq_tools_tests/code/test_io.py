#!/usr/bin/env python
import unittest
import sortseq_tools.io as io
import glob
from sortseq_tools import SortSeqError
from sortseq_tools import shutthefuckup



class TestLoading(unittest.TestCase):
    def setUp(self):
        self.input_dir = 'input/'
        self.output_dir = 'output/'

    def tearDown(self):
        pass

    @shutthefuckup
    def generic_test(self,test_name,function_str,file_names,allbad=False):
        """ 
        Standardizes tests for different dataframe loading functions.
        The argument function_str must have "%s" where file_name goes. 
        Example:
        generic_test('test_io_load_tagkey','io.load_tagkey("%s"),file_names)'
        """
        print '\nIn %s...'%test_name   

        # Make sure there are files to test
        self.assertTrue(len(file_names)>0)

        # For each file, run test
        for file_name in file_names:
            executable = lambda: eval(function_str%file_name)
            print '\t%s ='%file_name,
            if not allbad and any([c in file_name for c in \
                ['_good','_fix','_badio','_badtype']]):
                try:
                    df = executable()
                    self.assertTrue(df.shape[0]>=1)
                    # Write df
                    base_filename = file_name.split('/')[-1]
                    io.write(df,self.output_dir+'loaded_'+base_filename)
                    print 'good.'
                except:
                    print 'bad (ERROR).'
                    raise

            elif allbad or ('_bad' in file_name):
                try:
                    self.assertRaises(SortSeqError,executable)
                    print 'bad.'
                except:
                    print 'good (ERROR).'
                    raise
            else:
                print 'what should I expect? (ERROR)'
                raise
        print '\tDone.'


    def test_io_load_tagkey(self):
        """ Test io.load_tagkey
        """
        
        test_name = 'test_io_load_tagkey'
        function_str = 'io.load_tagkey("%s")'
        file_names = glob.glob(self.input_dir+'tagkey_*') 
        self.generic_test(test_name,function_str,file_names)


    def test_io_load_filelist(self):
        """ Test io.load_filelist
        """

        test_name = 'test_io_load_filelist'
        function_str = 'io.load_filelist("%s")'
        file_names = glob.glob(self.input_dir+'files_*') 
        self.generic_test(test_name,function_str,file_names)


    def test_io_load_model(self):
        """ Test io.load_model
        """

        test_name = 'test_io_load_model'
        function_str = 'io.load_model("%s")'
        file_names = glob.glob(self.input_dir+'model_*') 
        self.generic_test(test_name,function_str,file_names)


    def test_io_load_dataset_txt(self):
        """ Test io.load_dataset
        """

        test_name = 'test_io_load_dataset_txt'
        function_str = 'io.load_dataset("%s")'
        file_names = glob.glob(self.input_dir+'seq_*.txt') 
        self.generic_test(test_name,function_str,file_names)

    def test_io_load_dataset_raw_dna(self):
        """ Test io.load_dataset on raw sequence files
        """

        test_name = 'test_io_load_dataset_txt'
        function_str = 'io.load_dataset("%s",file_type="raw",seq_type="dna")'
        file_names = glob.glob(self.input_dir+'seqraw_*.txt') 
        self.generic_test(test_name,function_str,file_names)


    def test_io_load_dataset_fasta_dna(self):
        """ Test io.load_dataset( . ,"fasta")
        """

        test_name = 'test_io_load_dataset_fasta'
        function_str = 'io.load_dataset("%s",file_type="fasta",seq_type="dna")'
        file_names = glob.glob(self.input_dir+'seq_*.fasta') 
        self.generic_test(test_name,function_str,file_names)

    def test_io_load_dataset_fasta_dna_allbad(self):
        """ Test io.load_dataset( . ,"fasta")
        """

        test_name = 'test_io_load_dataset_fasta'
        function_str = 'io.load_dataset("%s",file_type="fasta",seq_type="dna")'
        file_names = glob.glob(self.input_dir+'seqrna_*.fasta') 
        self.generic_test(test_name,function_str,file_names,allbad=True)

    def test_io_load_dataset_fasta_rna_allbad(self):
        """ Test io.load_dataset( . ,"fasta")
        """

        test_name = 'test_io_load_dataset_fasta'
        function_str = 'io.load_dataset("%s",file_type="fasta",seq_type="rna")'
        file_names = glob.glob(self.input_dir+'seq_*.fasta') 
        self.generic_test(test_name,function_str,file_names,allbad=True)


    def test_io_load_dataset_fasta_protein_allbad(self):
        """ Test io.load_dataset( . ,"fasta")
        """

        test_name = 'test_io_load_dataset_fasta'
        function_str = \
            'io.load_dataset("%s",file_type="fasta",seq_type="protein")'
        file_names = glob.glob(self.input_dir+'seqrna_*.fasta') 
        self.generic_test(test_name,function_str,file_names,allbad=True)


    def test_io_load_dataset_fasta_rna(self):
        """ Test io.load_dataset( . ,"fasta",seq_type="rna")
        """

        test_name = 'test_io_load_dataset_fasta'
        function_str = 'io.load_dataset("%s",file_type="fasta",seq_type="rna")'
        file_names = glob.glob(self.input_dir+'seqrna_*.fasta') 
        self.generic_test(test_name,function_str,file_names)

    def test_io_load_dataset_fasta_protein(self):
        """ Test io.load_dataset( . ,"fasta",seq_type="pro")
        """

        test_name = 'test_io_load_dataset_fasta'
        function_str = \
            'io.load_dataset("%s",file_type="fasta",seq_type="protein")'
        file_names = glob.glob(self.input_dir+'seqpro_*.fasta') 
        self.generic_test(test_name,function_str,file_names)


    def test_io_load_dataset_fastq(self):
        """ Test io.load_dataset( . ,"fastq")
        """

        test_name = 'test_io_load_dataset_fastq'
        function_str = 'io.load_dataset("%s",file_type="fastq",seq_type="dna")'
        file_names = glob.glob(self.input_dir+'seq_*.fastq') 
        self.generic_test(test_name,function_str,file_names)

    def tearDown(self):
        pass
if __name__ == '__main__':
    unittest.main()
		
			
