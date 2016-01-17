#!/usr/bin/env python
import unittest
import sst.io
import glob

class Tests(unittest.TestCase):
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_io_load_model(self):
        """ Test the ability of sst.io.load to load correct model files and reject incorrect model files
        """

        # Verify loading of good files
        seq_files = glob.glob('data/model_good*')
        for file_name in seq_files:
            df = sst.io.load(file_name)

        # Verify TypeError for bad files
        seq_files = glob.glob('data/model_bad*')
        for file_name in seq_files:
            try:
                self.assertRaises(TypeError,\
                    lambda: sst.io.load(file_name))
            except:
                print 'This should have failed: sst.io.load("%s")'%file_name
                raise


    def test_io_load_seq_txt(self):
        """ Test the ability of sst.io.load to load correct seq text files and reject incorrect seq text files
        """
        # Verify loading of good files
        seq_files = glob.glob('data/seq_good*.txt')
        for file_name in seq_files:
            df = sst.io.load(file_name)

        # Verify TypeError for bad files
        seq_files = glob.glob('data/seq_bad*.txt')
        for file_name in seq_files:
            try:
                self.assertRaises(TypeError,\
                    lambda: sst.io.load(file_name))
            except:
                print 'This should have failed: sst.io.load("%s")'%file_name
                raise


    def test_io_load_fasta(self):
        """ Test the ability of sst.io.load to load correct fasta files and reject incorrect fasta files
        """

        # Verify loading of good files
        seq_files = glob.glob('data/seq_good*.fasta')
        for file_name in seq_files:
            try:
                df = sst.io.load(file_name,file_type="fasta")
            except:
                print 'This should have succeeded: sst.io.load("%s",file_type="fasta")'%file_name
                raise

        # Verify TypeError for bad files
        seq_files = glob.glob('data/seq_bad*.fasta')
        for file_name in seq_files:
            try:
                self.assertRaises(TypeError,\
                    lambda: sst.io.load(file_name),file_type="fasta")
            except:
                print 'This should have failed: sst.io.load("%s",file_type="fasta")'%file_name
                raise


    def test_io_load_fastq(self):
        """ Test the ability of sst.io.load to load correct fastq files and reject incorrect fastq files
        """

        # Verify loading of good files
        seq_files = glob.glob('data/seq_good*.fastq')
        for file_name in seq_files:
            df = sst.io.load(file_name,file_type="fastq")

        # Verify TypeError for bad files
        seq_files = glob.glob('data/seq_bad_*.fastq')
        for file_name in seq_files:
            try:
                self.assertRaises(TypeError,\
                    lambda: sst.io.load(file_name),file_type="fastq")
            except:
                print 'This should have failed: sst.io.load("%s",file_type="fastq")'%file_name
                raise

    def tearDown(self):
        pass
if __name__ == '__main__':
    unittest.main()
		
			
