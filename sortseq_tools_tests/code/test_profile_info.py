#!/usr/bin/env python
import unittest
import sortseq_tools.io as io
import sortseq_tools.qc as qc
import sortseq_tools.profile_info as profile_info
import glob
from sortseq_tools import SortSeqError
from sortseq_tools import shutthefuckup

class Tests(unittest.TestCase):
    def setUp(self):
        self.input_dir = 'input/'
        self.output_dir = 'output/'


    def tearDown(self):
        pass

    @shutthefuckup
    def test_profile_info(self):
        """ Test the ability of sortseq_tools.profile_info to compute mutation rates based on total count values
        """

        print '\nIn test_profile_info...'
        file_names = glob.glob(self.input_dir+'dataset_*.txt')
        for err in [True,False]:
            for file_name in file_names:
                print '\t%s, err=%s ='%(file_name,str(err)),
                description = file_name.split('_')[-1].split('.')[0]
                executable = lambda: \
                    profile_info.main(io.load_dataset(file_name),err=err)

                # If good, then profile_info.main should produce a valid df
                if '_good' in file_name:
                    try:
                        df = executable()
                        qc.validate_profile_info(df)
                        out_file = self.output_dir+\
                            'profile_info_%s_err_%s.txt'%(description,str(err))
                        io.write(df,out_file)
                        io.load_profile_info(out_file)
                        print 'good.'
                    except:
                        print 'bad (ERROR).'
                        raise

                # If bad, then profile_info.main should raise SortSeqError
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
		
			
