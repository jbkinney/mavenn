#!/usr/bin/env python
import unittest
import sst.io
import glob

class Tests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_bad_seq_file_io(self):
        """ Test the ability of sst.io.load to reject invalid files
        """
        bad_seq_files = glob.glob('data/seq*_bad_*.txt')

        for file_name in bad_seq_files:

            try:
                self.assertRaises(TypeError,\
                    lambda: sst.io.load(file_name))
            except:
                print 'Test failed while loading %s'%file_name
                raise

    def tearDown(self):
        pass
if __name__ == '__main__':
    unittest.main()
		
			
