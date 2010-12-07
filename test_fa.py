from glgm import *
from itertools import product
import time, timeit
from numpy import (sin, cos, amax)
from numpy.testing import (assert_array_almost_equal,
                           assert_almost_equal)


def testFA(d = 10, N = 5000, k = 4, min_iter_nr = 20):
    """
    The present test code comes from Modular toolkit for Data Processing (MDP) 
    source file test_FANode.py, slightly adapted to work with glgm.fa.
    """

    print "Input: dim, samples nr = %d, %d - Output: latent factors nr = %d" % (d, N, k)
    mu = uniform(1, d)*3.+2.
    sigma = uniform((d,))*0.01
    A = normal(size=(k,d))

    # latent variables
    y = normal(0., 1., size=(N, k))
    # observations
    noise = normal(0., 1., size=(N, d)) * sigma
    x = dot(y, A) + mu + noise
     
    for _b, _n in product((True, False), 
                          (min_iter_nr, min_iter_nr*2, min_iter_nr*3)):
        t_start = time.time()
        FA = fa(x.T, k = k)
        FA.InferandLearn(max_iter_nr = _n, inferwithLemma = _b)
        print "Model FA(max_iter_nr=%d, inferwithLemma=%s) learned in %.5f seconds" \
            % (_n, str(_b), time.time() - t_start)

        # compare estimates to real parameters
        assert_array_almost_equal(FA.mu_y(), mean(x, axis=0).reshape(d, 1), 5)
        assert_array_almost_equal(FA.R, std(noise, axis=0)**2, 2)
        
        # FA finds C only up to a rotation. here we verify that the
        # C and its estimation span the same subspace
        CC = concatenate((A,FA.C.T),axis=0)
        u,s,vh = svd(CC)
        assert sum(s / max(s) > 1e-2) == k, 'C and its estimation do not span the same subspace'
        
        y = FA.get_expected_latent(x.T)
        xi = FA.infer_observed()
        xi = FA.infer_observed(noised = True)
        xn = FA.get_new_observed(100)
        xn = FA.get_new_observed(y)
        xn = FA.get_new_observed(y, noised = True)
        xn = FA.get_new_observed(y, noised = True, centered = True)

        # test that noise has the right mean and variance
        est = FA.get_new_observed(zeros((k, N)), noised = True)
        est -= FA.mu_y()
        assert_array_almost_equal(diag(cov(est, rowvar=1)), FA.R, 3)
        #assert_almost_equal(amax(abs(mean(est, axis=1)), axis=None), 0., 3)

        #est = FA.get_new_observed(100000).T
        #assert_array_almost_equal_diff(cov(est, rowvar=0), multiply(fa.A, fa.A.T), 1)
    


if __name__ == '__main__':
    
    testFA()
    testFA(k=3)
    testFA(30, 10000, 4, 100)
    #testFA(50, 10000, 7, 200)
    #testFA(100, 20000, 10, 100)
