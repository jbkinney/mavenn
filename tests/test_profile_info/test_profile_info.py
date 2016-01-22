#!/usr/bin/env python
import unittest
import sst.io as io
import sst.qc as qc
import sst.profile_info as profile_info
import glob

class Tests(unittest.TestCase):
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_profile_info(self):
        """ Test the ability of sst.profile_info to compute mutation rates based on total count values
        """

        print '\nIn test_profile_info...'
        library_files = glob.glob('library_*.txt')
        library_files += glob.glob('dataset_*.txt')
        for err in [True,False]:
            for file_name in library_files:
                print '\t%s, err=%s ='%(file_name,str(err)),
                description = file_name.split('_')[-1].split('.')[0]
                executable = lambda: \
                    profile_info.main(io.load_dataset(file_name),err=err)

                # If good, then profile_info.main should produce a valid df
                if '_good' in file_name:
                    try:
                        df = executable()
                        qc.validate_profile_info(df)
                        out_file = 'profile_info_%s_err_%s.txt'%\
                            (description,str(err))
                        io.write(df,out_file)
                        io.load_profile_info(out_file)
                        print 'good.'
                    except:
                        print 'bad (ERROR).'
                        raise

                # If bad, then profile_info.main should raise TypeError
                elif '_bad' in file_name:
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
		
			
