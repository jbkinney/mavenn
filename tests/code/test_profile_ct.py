#!/usr/bin/env python
import unittest
import sortseq_tools.io as io
import sortseq_tools.qc as qc
import sortseq_tools.profile_ct as profile_ct
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
    def test_profile_ct_totalcounts(self):
        """ Test the ability of sortseq_tools.profile_ct to count frequencies based on total count values
        """

        print '\nIn test_profile_ct_totalcounts...'
        library_files = glob.glob(self.input_dir+'library_*.txt')
        library_files += glob.glob(self.input_dir+'dataset_*.txt')
        for file_name in library_files:
            print '\t%s ='%file_name,
            description = file_name.split('_')[-1].split('.')[0]
            executable = lambda: profile_ct.main(io.load_dataset(file_name))

            # If good, then profile_ct.main should produce a valid df
            if '_good' in file_name:
                try:
                    df = executable()
                    qc.validate_profile_ct(df)
                    out_file = self.output_dir+\
                        'profile_ct_total_%s.txt'%description
                    io.write(df,out_file)
                    io.load_profile_ct(out_file)
                    print 'good.'
                except:
                    print 'bad (ERROR).'
                    raise

            # If bad, then profile_ct.main should raise SortSeqError
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

    @shutthefuckup
    def test_profile_ct_bincounts(self):
        """ Test the ability of sortseq_tools.profile_ct to count frequencies
        """

        print '\nIn test_profile_ct_bincounts...'
        library_files = glob.glob(self.input_dir+'library_*.txt')
        library_files += glob.glob(self.input_dir+'dataset_*.txt')
        good_bin_num = 2
        bad_bin_num = 5
        for file_name in library_files:
            print '\t%s ='%file_name,
            description = file_name.split('_')[-1].split('.')[0]
            executable = lambda:\
                profile_ct.main(io.load_dataset(file_name),bin=good_bin_num)
            print '(bin=%d)'%good_bin_num,

            # If bad or library, then profile_ct.main should raise SortSeqError
            if ('_bad' in file_name) or ('library' in file_name):
                try:
                    self.assertRaises(SortSeqError,executable)
                    print 'badtype,',
                except:
                    print 'good (ERROR).'
                    raise

            # If good, then profile_ct.main should produce a valid df
            elif ('_good' in file_name) or ('dataset' in file_name):
                try:
                    df = executable()
                    qc.validate_profile_ct(df)
                    out_file = self.output_dir+\
                        'profile_ct_bin_%s.txt'%description
                    io.write(df,out_file)
                    io.load_profile_ct(out_file)
                    print 'good,',

                except:
                    print 'bad (ERROR).'
                    raise

            # There are no other options
            else:
                raise SortSeqError('Unrecognized class of file_name.')

            # Should always raise an error if bin num is too large
            executable = lambda:\
                profile_ct.main(io.load_dataset(file_name),bin=bad_bin_num)
            print '(bin=%d)'%bad_bin_num,
            try:
                self.assertRaises(SortSeqError,executable)
                print 'badtype.'
            except:
                print 'good (ERROR).'
                raise

    @shutthefuckup
    def test_profile_ct_seqslicing(self):
        """ Test the ability of sortseq_tools.profile_ct to slice sequences properly, and to raise the correct errors
        """

        print '\nIn test_profile_ct_seqslicing...'
        library_files = glob.glob(self.input_dir+'library_*.txt')
        library_files += glob.glob(self.input_dir+'dataset_*.txt')
        for file_name in library_files:
            print '\t%s ='%file_name,
            description = file_name.split('_')[-1].split('.')[0]
            executable_good1 =\
                lambda: profile_ct.main(io.load_dataset(file_name),\
                    start=2,end=10)
            executable_good2 =\
                lambda: profile_ct.main(io.load_dataset(file_name),\
                    start=2)
            executable_good3 =\
                lambda: profile_ct.main(io.load_dataset(file_name),\
                    end=2)
            executable_nopro =\
                lambda: profile_ct.main(io.load_dataset(file_name),\
                    start=50,end=60)
            executable_bad1 =\
                lambda: profile_ct.main(io.load_dataset(file_name),\
                    start=-1)
            executable_bad2 =\
                lambda: profile_ct.main(io.load_dataset(file_name),\
                    end=100)
            executable_bad3 =\
                lambda: profile_ct.main(io.load_dataset(file_name),\
                    start=20,end=10)

            # If good, then sequences will be valid
            if 'good' in file_name:
                try:
                    df = executable_good1()
                    io.write(df,self.output_dir+\
                        'profile_ct_splice2-10_%s.txt'%description)
                    executable_good2()
                    executable_good3()
                    self.assertRaises(SortSeqError,executable_bad1)
                    self.assertRaises(SortSeqError,executable_bad2)
                    self.assertRaises(SortSeqError,executable_bad3)
                    if '_pro' in file_name:
                        self.assertRaises(SortSeqError,executable_nopro)
                    else:
                        df = executable_nopro()
                    print 'ok.'
                except:
                    print 'ok (ERROR).'
                    raise

            # If bad, then profile_ct.main should raise SortSeqError
            elif '_bad' in file_name:
                try:
                    self.assertRaises(SortSeqError,executable_good1)
                    self.assertRaises(SortSeqError,executable_good2)
                    self.assertRaises(SortSeqError,executable_good3)
                    self.assertRaises(SortSeqError,executable_nopro)
                    self.assertRaises(SortSeqError,executable_bad1)
                    self.assertRaises(SortSeqError,executable_bad2)
                    self.assertRaises(SortSeqError,executable_bad3)
                    print 'ok.'
                except:
                    print 'not ok (ERROR).'
                    raise

            # There are no other options
            else:
                raise SortSeqError('Unrecognized class of file_name.')


if __name__ == '__main__':
    unittest.main()
		
			
