#!/usr/bin/env python
import unittest
import sst.io as io
import sst.qc as qc
import sst.preprocess as preprocess
import glob

class Tests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_preprocess(self):
        """ Test the ability of sst.preprocess to collate data in multiple sequence files
        """

        print '\nIn test_preprocess...'
        filelist_files = glob.glob('files_*.txt')
        for file_name in filelist_files:
            print '\t%s ='%file_name,
            description = file_name.split('_')[-1].split('.')[0]

            # All filelists should be valid
            try:
                filelist_df = io.load_filelist(file_name)
            except:
                print 'ERROR: io.load_filelist should have succeeded.'
                raise
            executable = lambda: preprocess.main(filelist_df)

            # If _good_, then preprocess.main should produce a valid df
            if '_good' in file_name:
                try:
                    df = executable()
                    qc.validate_dataset(df)
                    out_file = 'dataset_%s.txt'%description
                    io.write(df,out_file)       # Test write
                    io.load_dataset(out_file)   # Test loading
                    print 'good.'
                except:
                    print 'bad (ERROR).'
                    raise
            # If _badio_, then preprocess.main should raise IOError
            elif '_badio' in file_name:
                try:
                    self.assertRaises(IOError,executable)
                    print 'badio.'
                except:
                    print 'good (ERROR).'
                    raise

            # If _badtype_, then preprocess.main should raise TypeError
            elif '_badtype' in file_name:
                try:
                    self.assertRaises(TypeError,executable)
                    print 'badtype.'
                except:
                    print 'good (ERROR).'
                    raise

            # There are no other options
            else:
                raise TypeError('Unrecognized class of file_name.')



if __name__ == '__main__':
    unittest.main()
		
			
