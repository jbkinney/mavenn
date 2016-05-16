#!/usr/bin/env python
import unittest
import sortseq_tools.io as io
import glob
import numpy as np
import sortseq_tools.info
from sortseq_tools import SortSeqError
from sortseq_tools import shutthefuckup


class TestLoading(unittest.TestCase):

    @shutthefuckup
    def test_info_estimate_entropy(self):
        """
        Performs test on info.estimate_entropy
        """

        print '\nIn test_info_estimate_entropy...'

        methods = ['naive','nsb']

        # Prob distributions that should produce valid entropies
        good_ns_dict = {
            'small':100*np.arange(1,3),
            'large':100*np.arange(1,10),
            'somezeros':100*np.array([0,0,1]),
            '2d':100*np.ones([5,5]),
            '3d':100*np.ones([3,3,3])
        }

        # Verify that good prob dists produce good entropies
        for key in good_ns_dict.keys():
            ns = good_ns_dict[key]
            for err in [True,False]:
                for method in methods:
                    print '\tTesting %s with method=%s, err=%s: '\
                        %(key,method,str(err)),
                    try:
                        if err:
                            ent, ent_err = sortseq_tools.info.estimate_entropy(ns,\
                                    method=method,err=True)
                            self.assertTrue(ent_err >= 0.0)
                        else:
                            ent = sortseq_tools.info.estimate_entropy(ns,\
                                    method=method,err=False)

                        # Naive estimates should always be nonnegative
                        if method=='naive':
                            self.assertTrue(ent >= 0.0)

                        if err:
                            print 'good: ent == %f +- %f bits'%(ent,ent_err)
                        else:
                            print 'good: ent == %f bits'%ent
                    except:
                        print 'bad (ERROR).'
                        raise

        # Prob distributions that should raise errors
        bad_ns_dict = {
            'somenegative':range(-5,5),
            'allzero':np.zeros(10),
            'empty':[],
            'string':'hi there'
        }

        # Verify that bad prob dists raise errors
        for key in bad_ns_dict.keys():
            ns = bad_ns_dict[key]
            for err in [True,False]:
                for method in methods:
                    print '\tTesting %s with method=%s, err=%s: '\
                        %(key,method,str(err)),
                    executable = lambda: \
                        sortseq_tools.info.estimate_entropy(ns,method=method,err=err)
                    try:
                        self.assertRaises(SortSeqError,executable)
                        print 'bad.'
                    except:
                        print 'good (ERROR).'
                        raise
        print '\tDone.'

    @shutthefuckup
    def test_info_estimate_mutualinfo(self):
        """
        Performs test on info.estimate_mutualinfo
        """

        print '\nIn test_info_estimate_mutualinfo...'

        methods = ['naive','tpm','nsb']

        # Prob distributions that should produce valid mis
        good_counts_dict = {
            'small':100*np.array([[5,3],[3,5]]),
            'large':100*np.random.rand(5,5),
            'mi=0':100*np.ones([5,5])
        }

        # Verify that good prob dists produce good mis
        for key in good_counts_dict.keys():
            counts = good_counts_dict[key]
            for err in [True,False]:
                for method in methods:
                    print '\tTesting %s with method=%s, err=%s: '\
                        %(key,method,str(err)),
                    try:
                        if err:
                            mi, mi_err = sortseq_tools.info.estimate_mutualinfo(counts,\
                                    method=method,err=True)
                            self.assertTrue(mi_err >= 0.0)
                        else:
                            mi = sortseq_tools.info.estimate_mutualinfo(counts,\
                                    method=method,err=False)

                        # Naive estimates should always be nonnegative
                        if method=='naive':
                            self.assertTrue(mi >= 0.0)

                        if key=='mi=0' and method=='naive':
                            self.assertTrue(mi==0.0)
                            print '(mi==0)',

                        if err:
                            print 'good: mi == %f +- %f bits'%(mi,mi_err)
                        else:
                            print 'good: mi == %f bits'%mi
                    except:
                        print 'bad (ERROR).'
                        raise

        # Prob distributions that should raise errors
        bad_counts_dict = {
            '1d':[1, 2, 3],
            'allzero':np.zeros([5,5]),
            'empty':[],
            'string':'hi there',
            'somenegative':-np.ones([5,5])
        }

        # Verify that bad prob dists raise errors
        for key in bad_counts_dict.keys():
            counts = bad_counts_dict[key]
            for err in [True,False]:
                for method in methods:
                    print '\tTesting %s with method=%s, err=%s: '\
                        %(key,method,str(err)),
                    executable = lambda: \
                        sortseq_tools.info.estimate_mutualinfo(counts,method=method)
                    try:
                        self.assertRaises(SortSeqError,executable)
                        print 'bad.'
                    except:
                        print 'good (ERROR).'
                        raise
        print '\tDone.'


if __name__ == '__main__':
    unittest.main()
		
			
