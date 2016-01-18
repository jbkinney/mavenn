#!/usr/bin/env python
import unittest
import sst.io
import glob

class Tests(unittest.TestCase):
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_io_load_tagkey(self):
        """ Test the ability of sst.io.load_filelist to load correct model files and reject incorrect model files
        """

        # Verify loading of good files
        file_names = glob.glob('tagkey_good*')
        file_names += glob.glob('tagkey_fix*')
        for file_name in file_names:
            try:
                df = sst.io.load_tagkey(file_name)
                self.assertTrue(df.shape[1]==2)
                self.assertTrue(df.shape[0]>=1)
            except:
                print 'This should have succeeded: sst.io.load_tagkey("%s")'%file_name
                raise

        # Verify TypeError for bad files
        file_names = glob.glob('tagkey_bad*')
        for file_name in file_names:
            try:
                self.assertRaises(TypeError,\
                    lambda: sst.io.load_filelist(file_name))
            except:
                print 'This should have failed: sst.io.load_tagkey("%s")'%file_name
                raise


    def test_io_load_filelist(self):
        """ Test the ability of sst.io.load_filelist to load correct model files and reject incorrect model files
        """

        # Verify loading of good files
        file_names = glob.glob('files_good*')
        file_names += glob.glob('files_fix*')
        for file_name in file_names:
            try:
                df = sst.io.load_filelist(file_name)
                self.assertTrue(df.shape[1]==2)
                self.assertTrue(df.shape[0]>=1)
            except:
                print 'This should have succeeded: sst.io.load_filelist("%s")'%file_name
                raise

        # Verify TypeError for bad files
        file_names = glob.glob('files_bad*')
        for file_name in file_names:
            try:
                self.assertRaises(TypeError,\
                    lambda: sst.io.load_filelist(file_name))
            except:
                print 'This should have failed: sst.io.load_filelist("%s")'%file_name
                raise

    def test_io_load_model(self):
        """ Test the ability of sst.io.load_model to load correct model files and reject incorrect model files
        """

        # Verify loading of good files
        file_names = glob.glob('model_good*')
        file_names += glob.glob('model_fix*')
        for file_name in file_names:
            try:
                df = sst.io.load_model(file_name)
                self.assertTrue(df.shape[1] in [5,17,21,401])
                self.assertTrue(df.shape[0]>=1)
            except:
                print 'This should have succeeded: sst.io.load_model("%s")'%file_name
                raise

        # Verify TypeError for bad files
        file_names = glob.glob('model_bad*')
        for file_name in file_names:
            try:
                self.assertRaises(TypeError,\
                    lambda: sst.io.load_model(file_name))
            except:
                print 'This should have failed: sst.io.load_model("%s")'%file_name
                raise


    def test_io_load_dataset_txt(self):
        """ Test the ability of sst.io.load_dataset to load correct seq text files and reject incorrect seq text files
        """
        # Verify loading of good files
        file_names = glob.glob('seq_good*.txt')
        file_names += glob.glob('seq_fix*')
        for file_name in file_names:
            try:
                df = sst.io.load_dataset(file_name)
                self.assertTrue(df.shape[1]>=2)
                self.assertTrue(df.shape[0]>=1)
            except:
                print 'This should have succeeded: sst.io.load_dataset("%s")'%file_name
                raise

        # Verify TypeError for bad files
        file_names = glob.glob('seq_bad*.txt')
        for file_name in file_names:
            try:
                self.assertRaises(TypeError,\
                    lambda: sst.io.load_dataset(file_name))
            except:
                print 'This should have failed: sst.io.load_dataset("%s")'%file_name
                raise


    def test_io_load_dataaset_fasta(self):
        """ Test the ability of sst.io.load_dataset to load correct fasta files and reject incorrect fasta files
        """

        # Verify loading of good files
        file_names = glob.glob('seq_good*.fasta')
        file_names += glob.glob('seq_fix*.fasta')
        for file_name in file_names:
            try:
                df = sst.io.load_dataset(file_name,file_type="fasta")
                self.assertTrue(df.shape[1]>=2)
                self.assertTrue(df.shape[0]>=1)
            except:
                print 'This should have succeeded: sst.io.load_dataset("%s",file_type="fasta")'%file_name
                raise

        # Verify TypeError for bad files
        file_names = glob.glob('seq_bad*.fasta')
        for file_name in file_names:
            try:
                self.assertRaises(TypeError,\
                    lambda: sst.io.load_dataset(file_name),file_type="fasta")
            except:
                print 'This should have failed: sst.io.load_dataset("%s",file_type="fasta")'%file_name
                raise


    def test_io_load_dataset_fastq(self):
        """ Test the ability of sst.io.load_dataset to load correct fastq files and reject incorrect fastq files
        """

        # Verify loading of good files
        file_names = glob.glob('seq_good*.fastq')
        file_names += glob.glob('seq_fix*.fastq')
        for file_name in file_names:
            try:
                df = sst.io.load_dataset(file_name,file_type="fastq")
                self.assertTrue(df.shape[1]>=2)
                self.assertTrue(df.shape[0]>=1)
            except:
                print 'This should have succeeded: sst.io.load_dataset("%s",file_type="fasta")'%file_name
                raise

        # Verify TypeError for bad files
        file_names = glob.glob('seq_bad_*.fastq')
        for file_name in file_names:
            try:
                self.assertRaises(TypeError,\
                    lambda: sst.io.load_dataset(file_name),file_type="fastq")
            except:
                print 'This should have failed: sst.io.load_dataset("%s",file_type="fastq")'%file_name
                raise

    def tearDown(self):
        pass
if __name__ == '__main__':
    unittest.main()
		
			
