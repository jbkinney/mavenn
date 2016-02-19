#!/usr/bin/env python
import unittest
import sortseq_tools.io as io
import sortseq_tools.qc as qc
import sortseq_tools.preprocess as preprocess
import glob
from sortseq_tools import SortSeqError
from sortseq_tools import shutthefuckup

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.input_dir = 'input/'
        self.output_dir = 'output/'

    def tearDown(self):
        pass

    
    #@shutthefuckup
    def test_preprocess(self):
        """ Test the ability of sortseq_tools.preprocess to collate data in multiple sequence files
        """

        print '\nIn test_preprocess...'
        file_names = glob.glob(self.input_dir+'files_*.txt')

        # Make sure there are files to test
        self.assertTrue(len(file_names)>0)

        for file_name in file_names:
            print '\t%s ='%file_name,
            description = file_name.split('_')[-1].split('.')[0]

            # If fasta or fastq, assume dna
            if ('fasta' in file_name) or ('fastq' in file_name):
                seq_type = 'dna'
            else:
                seq_type = None

            executable = lambda: preprocess.main(io.load_filelist(file_name),indir=self.input_dir, seq_type=seq_type)

            # If _good_, then preprocess.main should produce a valid df
            if ('_good' in file_name) or ('_fix' in file_name):
                try:
                    df = executable()
                    qc.validate_dataset(df)
                    out_file = self.output_dir+'dataset_%s.txt'%description
                    io.write(df,out_file)       # Test write
                    io.load_dataset(out_file)   # Test loading
                    print 'good.'
                except:
                    print 'bad (ERROR).'
                    raise

            # If _bad, then preprocess.main should raise SortSeqError
            elif '_bad' in file_name:
                try:
                    self.assertRaises(SortSeqError,executable)
                    print 'badtype.'
                except:
                    print 'good (ERROR).'
                    raise

            # There are no other options
            else:
                raise SortSeqError('Unrecognized class of file_name.')



if __name__ == '__main__':
    unittest.main()
		
			
