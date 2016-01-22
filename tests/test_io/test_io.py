#!/usr/bin/env python
import unittest
import sst.io as io
import glob


class TestLoading(unittest.TestCase):
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def generic_test(self,test_name,function_str,file_names):
        """ 
        Standardizes tests for different dataframe loading functions.
        The argument function_str must have "%s" where file_name goes. 
        Example:
        generic_test('test_io_load_tagkey','io.load_tagkey("%s"),file_names)'
        """
        print '\nIn %s...'%test_name                 
        for file_name in file_names:
            executable = lambda: eval(function_str%file_name)
            print '\t%s ='%file_name,
            if '_good' in file_name or '_fix' in file_name:
                try:
                    df = executable()
                    self.assertTrue(df.shape[0]>=1)
                    print 'good.'
                except:
                    print 'bad (ERROR).'
                    raise

            elif '_bad' in file_name:
                try:
                    self.assertRaises(TypeError,executable)
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
        file_names = glob.glob('tagkey_*') 
        self.generic_test(test_name,function_str,file_names)


    def test_io_load_filelist(self):
        """ Test io.load_filelist
        """

        test_name = 'test_io_load_filelist'
        function_str = 'io.load_filelist("%s")'
        file_names = glob.glob('files_*') 
        self.generic_test(test_name,function_str,file_names)


    def test_io_load_model(self):
        """ Test io.load_model
        """

        test_name = 'test_io_load_model'
        function_str = 'io.load_model("%s")'
        file_names = glob.glob('model_*') 
        self.generic_test(test_name,function_str,file_names)


    def test_io_load_dataset_txt(self):
        """ Test io.load_dataset
        """

        test_name = 'test_io_load_dataset_txt'
        function_str = 'io.load_dataset("%s")'
        file_names = glob.glob('seq_*.txt') 
        self.generic_test(test_name,function_str,file_names)


    def test_io_load_dataset_fasta(self):
        """ Test io.load_dataset( . ,"fasta")
        """

        test_name = 'test_io_load_dataset_fasta'
        function_str = 'io.load_dataset("%s","fasta")'
        file_names = glob.glob('seq_*.fasta') 
        self.generic_test(test_name,function_str,file_names)


    def test_io_load_dataset_fastq(self):
        """ Test io.load_dataset( . ,"fastq")
        """

        test_name = 'test_io_load_dataset_fastq'
        function_str = 'io.load_dataset("%s","fastq")'
        file_names = glob.glob('seq_*.fastq') 
        self.generic_test(test_name,function_str,file_names)

    def tearDown(self):
        pass
if __name__ == '__main__':
    unittest.main()
		
			
