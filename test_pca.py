from glgm import *
from itertools import product
import time, timeit
from numpy import (sin, cos, amax)
from numpy.testing import (assert_array_almost_equal,
                           assert_almost_equal)


def testSPCA(d = 10, N = 5000, k = 4, min_iter_nr = 20):

    print "Input: dim, samples nr = %d, %d - Output: latent factors nr = %d" % (d, N, k)
    mu = uniform(1, d)*3.+2.
    sigma = uniform((d,))*0.01
    A = normal(size=(k,d))

    # latent variables
    y = normal(0., 1., size=(N, k))
    # observations
    noise = normal(0., 1., size=(N, d)) * sigma
    x = dot(y, A) + mu + noise
     
    # Testing SPCA (or PPCA)
    t_start = time.time()
    PCA = spca(x.T, k = k)
    PCA.InferandLearn(max_iter_nr = _n)
    print "SPCA learned in %.5f seconds" % (time.time() - t_start)
     

def testPCA(d = 10, N = 5000, k = 4, min_iter_nr = 20):
    """
    The present test code is based on source file test_PCANode.py, 
    from  Modular toolkit for Data Processing (MDP), slightly adapted for glgm.pca.
    NB MDP actually calculates PCA via eigenvalues, or SVD, or iteratively (via nipals) 
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
     
    # Testing PCA 
    for _b, _n in product((True, False), (min_iter_nr, )):
        t_start = time.time()
        PCA = pca(x.T, k = k)
        PCA.InferandLearn(max_iter_nr = _n, svd_on = _b)
        print "PCA(svd_on=%s, max_iter_nr=%d) learned in %.5f seconds" % (str(_b), _n, time.time() - t_start)
        print PCA.C
        print "-"*70


if __name__ == '__main__':
    
    #testSPCA(k=3)
    testPCA(k=3)
