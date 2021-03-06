try:
    import psyco
    psyco.full()
except ImportError: pass


__doc__ = """
The present module Generative Linear Gaussian Models (GLGM) implements in Python programming language
the following models based mainly on references nr.1 (see References below):

        x(t+1) = Ax(t) + w(t) = Ax(t) + w0,     w0 = N(0, Q)
        y(t)   = Cx(t) + v(t) = Cx(t) + v0,     v0 = N(0, R)
     where:
        x is a (k,1) vector of latent factors or state or causes (hidden) variables
        y is a (p,1) vector of observations
        A is a (k,k) matrix of states transition
        C is a (p,k) matrix of observation measurement or generative matrix
        w is a (k,1) vector of state evolution White Gaussian Noise (WGN)
        v is a (k,1) vector of observation evolution WGN
        w and v are statistically independent of each other and of x and y
        N stands for for Gaussian or Normal probability density function (pdf)

In particular the module aims to implement the following:
    - fa    (Factor Analysis)
    - ppca  (Probabilistic Principal Component Analysis)
    - pca   (Principal Component Analysis)
    - mog   (Mixture of Gaussians)
    - vq    (Vector Quantization)
    - k-means clustering
    - mofa  (Mixture of Factor Analyzers)
    - ica   (Independent Component Analysis)

References:
    1  Unifying Review of Linear Gaussian Models
        Roweis, Ghahramani
        Neural Computation 1999 11:2, 305-345.
    2  The EM algorithm for mixtures of factor analyzers
        Ghahramani, Hinton
        1997, Technical Report CRG-TR-96-1
        Dept. of Computer Science, University of Toronto, Toronto, Canada, MSS 1A4
    3  Max Welling's tutorials 
        (available @ http://www.ics.uci.edu/~welling/classnotes/classnotes.html)
    4  Book of Johnson and Wichern
    5  Book of Joreskog
    6  Book of Duda, Hart
    7  Book of MacKay
    8  Book of Golub, Van Loan
    9  Maximum Likelihood and Covariant Algorithms for Independent Component Analysis
        MacKay
    10 Solving inverse problems using an EM approach to density estimation
        Ghahramani
    11 Maximum likelihood and minimum classification error Factor Analysis for automatic speech recognition
        Saul, Rahim
    12 Finite mixture models
        Geoffrey J. McLachlan, David Peel [book of] 
    13 Numerical recipes: the art of scientific computing
        Press et al [book of]
    14 Unsupervised Classification with Non-Gaussian Mixture Models Using ICA
        Lee, Lewicki, Sejnowski
    15 EM Algorithms for PCA and SPCA
        Roweis
    16 A tutorial on Hidden Markov Models and selected applications in speech recognition
        Rabiner
    17 Maximum Likelihood Estimation for Multivariate Mixture Observations of Markov Chains
        Juang, Levinson, Sondhi   

NB Matrix x and y have both shape given by the tuple ('variables nr','samples nr').
"""

from numpy import (array, arange, dot, inner, outer, vdot, cov,
                   diag, ones, eye, zeros, argmax, nanargmax, 
                   mean, std, multiply, sum, product, sqrt,
                   log, abs, exp, power, hstack, vstack, append,
                   concatenate, pi, inf, amin, amax, empty,
                   tanh, any, isnan)
from scipy.linalg import (norm, inv, det, svd, solve, cholesky)
from numpy.random import (normal, randn, rand, multivariate_normal,
                          uniform)


class lm(object):
    """See details explained above"""

    y, n, p = _default_values_ = None, 0, None
    cumulate = True
    
    def __init__(self, y, *args, **kwargs):

        if not(hasattr(y, '__iter__')):
            raise AttributeError('Input has to an iterable')
        self.y = y
        self.p, self.n = y.shape
             
    def __call__(self, **kw):
        
        return self.InferandLearn(**kw)

    def InferandLearn(self, max_iter_nr = 20, **kwargs):
        """
        This method implement and run the EM algorithm, in order to:
        - learn model's parameter A, C, Q, R (learning or system identification)
        - estimate hidden states from observations (inference or filtering or smoothing)

        Inference and Learning, respectively in methods E and M, are overrided by subclasses.
        """
        
        if not isinstance(max_iter_nr, int):
            raise TypeError('The maximum number of iterations of the EM procedure must be an integer')
        if max_iter_nr <= 0:
            raise ValueError('The maximum number of iterations of the EM procedure must be positive')
        for kw, val in kwargs.iteritems():
            self.kw = val
        E_EM, M_EM  = self.Inference, self.Learning 
        logLH, Break = self.logLikelihood, self.break_condition
        self.logLH_delta = kwargs.get('logLH_delta', None)
        
        for iter_nr in xrange(max_iter_nr):
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # E step
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            E_EM()
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # log-LH
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            logLH()
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # M step
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            M_EM()
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Break condition of the for loop
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if Break(): break
        
        self.iter_nr = iter_nr
        self.trained = True

    def next(self, **kwargs):
        """Calling InferandLearn method to run only one iteration"""

        self.InferandLearn(max_iter_nr = 1, **kwargs)
    
    def Inference(self):
        """The E(xpectation) step of EM algorithm"""
        
        raise NotImplementedError('Method Inference not implemented in base class')

    def Learning(self):
        """The M(aximization) step of EM algorithm"""

        raise NotImplementedError('Method Learning not implemented in base classi')
    
    def break_condition(self): 
        """A method verifying an ad-hoc condition to exit from EM iter."""

        raise NotImplementedError('Method break_condition not implemented in base class')
    
    def logLikelihood(self):
        """The logLikelihood method for the given model"""

        raise NotImplementedError('Method logLikelihood not implemented in base class')
    
    def mu_y(self):
        
        return mean(self.y, axis = 1).reshape(self.p, 1)
    
    def cov_obs(self, cov_bias = 1):
        
        return cov(self.y, bias = cov_bias)
    
    def centered_input(self):
        
        return self.y - self.mu_y()

    def erase(self):
        
        self.y, self.n, self.p = lm._default_values_

    
class fa(lm):
    """
    Factor Analysis model of static data y.
    Model:
        A = 0     (because data are static in time)
        x = x0 = w0,        w0 = N(0, Q)
        y = y0 = Cx + v0,   v0 = N(0, R)
    then
        y ~ N(0, CQC.T + R) 
    and in order to solve any model degeneracy
        Q = I
        R is diagonal
    finally
        y   ~ N(0, C*C.T + R)
        x_y ~ N(beta*y, I-beta*C)   useful for the Inference task
        beta = C.T(C*C.T + R)**-1
    
    C is also called the factor loadings matrix,
    R's diagonal elements are the uniquenesses,
    v the sensor noise.
    
    Hint: apply Template Design Pattern to scale-down from fa
          to spca (or ppca), pca and whitening sub-classes.
    
    Based on ref.1
    """

    k = None
    
    def __init__(self, y, k):
        
        super(fa, self).__init__(y)
        if not isinstance(k, int):
            raise TypeError('k (the number of latent factors) must be an integer')
        if k <= 0:
            raise ValueError('k (the number of latent factors) must be positive')
        if k > self.p:
            raise ValueError('k (the number of latent factors) must not be greater than p (the number of observables)')
        self.k = k

        self.initialize()

    def initialize_Q(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Covariance matrix Q of hidden factors = I matrix
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.Q = eye(self.k)
    
    def initialize_C(self):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Following MDP init settings
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        scale = product(self.yyTdiag) ** (1./self.p)
        assert scale > 0, "Input covariance matrix is singular"
        self.C = normal(0, sqrt(scale / self.k), size = (self.p, self.k))
    
    def initialize_R(self, with_WN = False):
        
        self.R = self.yyTdiag
        if with_WN: self.R += randn(self.p)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # From pag.531 of ref.4 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ## self.R = (1-.5*self.p/self.k)*self.yyTdiag

    def initializelogLH(self):
        
        self.logLH = -inf
        self.deltalogLH = inf
        self.logLH_const = -.5 * self.p * log(2. * pi)
        self.logLH_tracks = []
        self.logLH__break = False

    def initialize(self):
        """Initialization step: C and other vars, get observed data covariance"""

        self.arangek, self.arangep = arange(self.k), arange(self.p)
        
        self.yyT = self.cov_obs()
        self.yyTdiag = self.yyT[self.arangep, self.arangep]
        self.initialize_C()
        self.initialize_R()
        self.initialize_Q()
        self.initializelogLH()

    def InferandLearn(self, max_iter_nr = 20, logLH_delta = 1e-3,
                      inferwithLemma = True, **kwargs):

        self.betaInferenceMethod = self.betaInferenceLemma \
                            if inferwithLemma else self.betaInference

        super(fa, self).InferandLearn(max_iter_nr = max_iter_nr,
                                      logLH_delta = logLH_delta)
        
    def break_condition(self):
        
        if -self.logLH_delta < self.deltalogLH < self.logLH_delta:
            self.logLH__break = True
            return True
        return False
    
    def betaInferenceLemma(self):

        C, CT, Rinv = self.C, self.C.T, self.R ** -1
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Applying the matrix inversion lemma
        # See: - pag.334 in ref.1
        #      - aka Sherman-Morrison-Woodbury formula at pag.50 
        #        in ref.8 (formula (2.1.4)
        #      - aka binomial inverse theorem at 
        #        http://en.wikipedia.org/wiki/Binomialinverse_theorem
        #      - or derived from matrix blockwise inversion as in 
        #        http://en.wikipedia.org/wiki/Invertible_matrix#Blockwiseinversion
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        beta_temp = multiply(CT, Rinv)
        beta = dot(beta_temp, C)
        beta[self.arangek, self.arangek] += 1.
        beta = -dot(C, dot(inv(beta), beta_temp))
        beta[self.arangep, self.arangep] += 1.
        self.logLH_temp = multiply(Rinv.reshape(self.p, 1), beta)
        # self.logLH_temp = Rinv * beta #multiply(Rinv, beta)
        self.beta = dot(beta_temp, beta)

    def betaInference(self):

        C, R = self.C, self.R
        CT = C.T
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Applying the classical method to invert beta
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        beta = dot(C, CT)
        beta[self.arangep, self.arangep] += R
        self.logLH_temp = beta         # = inv(beta)
        self.beta = dot(CT, inv(beta)) #beta)

    def Inference(self):
        """
        E step of EM algorithm
        Inference of sufficient statistic E(x|y) (here x_y)
        
        NB Computing beta via the matrix inversion lemma,
           in place of apply ordinary formulas, does not
           bring performances improvements.
        NB Following code is very inefficient: 
            beta.ravel()[arange(0, k**2, k) + arange(k)] += 1.
            beta.ravel()[arange(0, p**2, p) + arange(p)] += 1.
            self.V.ravel()[arange(0, k**2, k) + arange(k)] += 1.

        TODO Test if betaInferenceLemma gives performances advantages
             for (very) high p.
        """
        
        self.betaInferenceMethod()
        self.V = - dot(self.beta, self.C)
        self.V[self.arangek, self.arangek] += 1.

    def Learning_R(self):
        """
        Learning R in M step of EM algorithm
        It is necessary to learn R separately in order to give
        sub-classes able to override the present method
        """
        
        self.R = self.yyTdiag - sum(multiply(self.C, self.delta), axis=1) / self.n

    def Learning(self):
        """
        M step of EM algorithm
        Computing delta and gamma (ref.1 at pag.335).
        Learning and updating model's parameters C and R
        """
        
        delta = dot(self.yyT, self.beta.T)
        self.gamma = self.n * (dot(self.beta, delta) + self.V)
        delta *= self.n
        self.delta = delta
        self.C = dot(self.delta, inv(self.gamma))
        self.Learning_R()

    def logLikelihood(self, mu = None):
        """
        Log-likelihood (LogLH)
        This the LogLH per sample, apart the constant self.logLH_const and
        factor .5 both affecting very sligthly the logLH values convergence
        to a (local, hopefully global) maximum.
        """
        
        logLH_old = self.logLH
        _yyT =  self.yyT - dot(mu, mu.T) if mu != None else self.yyT 
        self.logLH = self.logLH_const - .5 * (-log(det(self.logLH_temp)) + \
                     vdot(_yyT, self.logLH_temp))
        self.deltalogLH = self.logLH - logLH_old
        self.logLH_tracks.append(self.logLH)

    def infer(self):
        
        return self.get_expected_latent(), self.infer_observed()

    def get_expected_latent(self, z):
        
        if not(self.trained): self.InferandLearn()
        return dot(self.beta, z - self.mu_y())
    
    #def infer_observed(self, noised = False):
    #    
    #    if not(self.trained): self.InferandLearn()
    #    inf_y = dot(self.C, self.y)) - self.mu_y()
    #    if noised:
    #        return inf_y + multivariate_normal(zeros(self.p),
    #                                           diag(self.R), self.n).T
    #    return inf_y
    
    def get_new_observed(self, input, noised = False):
        """input: nr. or latent samples corresponding to new observations in output"""

        if not(self.trained):
            self.InferandLearn()
        if isinstance(input, int):
            input = normal(size = (self.k, input))
        new_obs = dot(self.C, input) + self.mu_y()
        if noised:
            new_obs += multivariate_normal(zeros(self.p),
                                           diag(self.R), input.shape[1]).T
        return new_obs


class spca(fa):
    """
    Sensible (or Probabilistic) Principal Component Analysis. See ref.1 and 15
    """ 

    def initialize_R(self):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Covariance matrix R of observations = const*I matrix
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.R = ones(self.p) * mean(self.yyTdiag)

    def Learning_R(self):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Covariance matrix R of observations = const*I matrix
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.R = ones(self.p) * mean(self.yyTdiag - \
                        sum(multiply(self.C, self.delta), axis=1) / self.n)


class ppca(spca):
    """Alias for spca"""

    pass


class pca(fa):
    """
    EM algorithm for Principal Component Analysis. See ref.1 and 15.
    TODO: Check EM method in order to get results comparable with the svd's ones
    """
    
    def __init__(self, y, k = None):
        
        super(fa, self).__init__(y)
        if not isinstance(k, int):
            raise TypeError('k (the number of latent factors) must be an integer')
        if k <= 0:
            raise ValueError('k (the number of latent factors) must be positive')
        if k > self.p:
            raise ValueError('k (the number of latent factors) must not be greater than p (the number of observables)')
        self.k = k

        self.initialize()
        
    def initialize_C(self):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Following MDP init settings
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        scale = 1. #product(self.yyTdiag) ** (1./self.p)
        self.C = normal(0, sqrt(scale / self.k), size = (self.p, self.k))
        self.C = array([e / norm(e) for e in self.C.T]).T
    
    def initialize_R(self):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Covariance matrix R of observations = Zeros matrix
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.R = zeros(self.p)

    def Inference(self):
        
        self.betaInference()
    
    def betaInference(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Read note 10, pag.318, ref.1 about the fomula of beta
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        CT = self.C.T
        self.beta = dot(inv(dot(CT, self.C)), CT)

    def InferandLearn(self, max_iter_nr = 20, svd_on = True, **kwargs):
        
        self.V = zeros((self.k, self.k))
        
        if svd_on:
            self.svd_on = True
            self.svd()
        else:
            self.svd_on = False 
            lm.InferandLearn(self, max_iter_nr = max_iter_nr, **kwargs)
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # According to ref.15: "...The columns of C will span the
            # space of the first k principal components. (To compute
            # the corresponding eigenvectors and eigenvalues explicitly,
            # the data can be projected into this-dimensional subspace 
            # and an ordered orthogonal basis for the covariance in the
            # subspace can be constructed.)..."
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.projOrthogBasis()
            
    def projOrthogBasis(self):
        
        xlatent = self.get_expected_latent(self.y)
        U, s, V = svd(xlatent-xlatent.mean(axis=1).reshape(self.k, 1),
                      full_matrices = False)
        self._variances = s
        self._scores = (s.reshape(self.k, 1) * V)
        self._loadings = U
        self.C = dot(self.C, U)
        self.trained = True

    def Learning(self):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # M step of EM algorithm
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.delta = dot(self.yyT, self.beta.T)
        self.gamma = dot(self.beta, self.delta)
        self.C = dot(self.delta, inv(self.gamma))
    
    def logLikelihood(self): pass
    
    def Learning_R(self): pass

    def logLikelihood(self): pass
    
    def break_condition(self): pass #return self.lse() #based on Least Squares projection error
    
    def lse(self): pass

    def svd(self):
        """Finding Principal Components via SVD method"""
        
        U, s, V = svd(self.y-self.mu_y(), full_matrices = False)
        self._variances = s
        self._scores = (s[:self.k].reshape(self.k, 1) * V[:self.k,:])
        self._loadings = U
        self.C = U[:,:self.k]
        self.trained = True
       
    #def get_expected_latent(self): return dot(self.beta, self.y)
    #def get_latent(self): return dot(self.C.T, self.centered_input())


class whitening(pca):

    def InferandLearn(self, **kwargs):
        
        pca.InferandLearn(self, svd_on = True, **kwargs)
        self.C /= sqrt(self.VarSvd[:self.k]).reshape(1, self.k)


class mixture(lm):

    m, typePrior_mu = None, ''

    def initialize(self):
        """
        Initialization step
        - pi with random values sum up to unity
        - mu from a Uniform pdf in the range of min/max of input data
            or a Gaussian pdf with mean and var of input data
        - sigma with a random square matrix for each cluster
        """
    
        self.arangep = arange(self.p)
        self.yyT = self.cov_obs()
        self.yyTdiag = self.yyT[self.arangep, self.arangep]
        
        self.initialize_pi()
        self.initialize_mu()
        self.initialize_Resp()
        self.initialize_sigma()
        self.initializelogLH()

    def initialize_Resp(self):

        self.Resp = rand(self.m, self.n)
        self.normalizeResp()

    def Resppower(self): pass

    def normalizeResp(self):
        
        self.Resp /= self.Resp.sum(axis = 0)
    
    def initialize_pi(self):

        self.pi = rand(self.m)
        self.pi /= self.pi.sum()
    
    def pi_clusters(self):
        
        self.pi = self.Resp.sum(axis = 1) / self.n

    def Normalprior_mu(self, scale = 1):
        """Normal prior on cluster's centroids"""

        self.mu = multivariate_normal(self.mu_y().ravel(), \
                        self.cov_obs() / scale, self.m).reshape(self.m, self.p)

    def Uniformprior_mu(self):
        """Uniform prior on cluster's centroids"""
        
        _uniform = lambda i, j: uniform(i, j, self.m) 
        self.mu = array(map(_uniform, amin(self.y, axis=1), \
                            amax(self.y, axis=1))).reshape(self.m, self.p)

    def initialize_mu(self):
        """Following a Strategy DP"""
    
        if self.typePrior_mu == 'normal' or not(self.typePrior_mu):
            self.Normalprior_mu()
        if self.typePrior_mu == 'uniform':
            self.Uniformprior_mu()
        
    def initializelogLH(self):
        
        self.logLH = -inf
        self.deltalogLH = inf
        self.logLH_tracks = []
        self.logLH__break = False
        self.logLH_const = -.5 * self.p * log(2. * pi)

        # An help from already available Resp values
        self.LH_temp = empty((self.m, self.n))    
    
    def CrossProdFactory(self, inv_sigma):
        """Curried method in order to speed up the processing"""

        def CrossProdinner(z):
            return dot(dot(z, inv_sigma), z.T)
        return CrossProdinner
    
    def Inference(self):
        """E step of EM algorithm"""

        for Resp_i, pi, mu, sigma, LH in zip(self.Resp, self.pi, self.mu,
                                             self.sigma, self.LH_temp):
            yCent_i = self.y - mu.reshape(self.p, 1)
            try:
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # There's no need to multiply const_i by (2*pi)**self.p
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                const_i = abs(det(sigma)) ** -.5
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Currying CrossProdFactory with inv(self.sigma[i])
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                _CrossProd = self.CrossProdFactory(inv(sigma)) 
                Resp_i[:] = LH[:] = pi * const_i * \
                                    exp(-.5 * array(map(_CrossProd, yCent_i.T)))
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # If no data in the i-th cluster, fixing pi[i] to zero 
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            except linalg.LinAlgError:
                pi = 0.
                pass

        self.Resppower()
        self.normalizeResp()
        
    def Learning(self):
        """
        M step of EM algorithm
        Call learning sub-methods in the following order:
            I)   self.mu_clusters()
            II)  self.sigma_clusters()
            III) self.pi_clusters()
        NB It's mandatory to call self.mu_clusters() before
        self.sigma_clusters(), because the latter needs the
        terms just updated by self.mu_clusters. 
        """
        
        self.mu_clusters()
        self.sigma_clusters()
        self.pi_clusters()

    def logLikelihood(self):

        logLH_old = self.logLH
        """
        LH = 0.
        LHComp = []
        for pi, mu, sigma in zip(self.pi, self.mu, self.sigma):
            const_i = const * abs(det(sigma)) ** -.5
            inv_sigma = inv(sigma)
            _mu = mu.reshape(self.p, 1)
            LHComp.append(pi * vdot(self.yyT - dot(_mu, _mu.T), .5 * inv_sigma))
        self.logLH = sum(LHComp)
        """
        self.logLH = self.logLH_const + sum(map(log, self.LH_temp.sum(axis = 0))) / self.n

        self.deltalogLH = self.logLH - logLH_old
        self.logLH_tracks.append(self.logLH)

    def MAP(self):
        
        return argmax(self.Resp, axis = 0)

    def GetExpectedLatent(self):
        
        return self.MAP()
    
    # No (WN) noise added
    def InferObs(self):
        mp = self.MAP()
        return [self.y[:, mp == i] for i in xrange(self.m)]

    # No (WN) noise added
    def GetNewObs(self, centered = False): pass    
    
    def GetCompProb(self, obs):
        
        CompProb = []
        for pi, mu, sigma in zip(self.pi, self.mu, self.sigma):
            try:
                const_i = ((2*pi) ** self.p) * (abs(det(sigma)) ** -.5)
                _CrossProd = self.CrossProdFactory(inv(sigma))
                CompProb.append(pi * const_i * \
                                _CrossProd(obs - mu.reshape(self.p, 1)))
            except linalg.LinAlgError: CompProb.append(0.)
        return CompProb 

    def entropy(self):
        
        raise NotImplementedError


class mog(mixture):
    """Based mainly on ref.3, and also on ref.1, 6 and 7"""

    def __init__(self, y, m, typePrior_mu = ''):

        super(mog, self).__init__(y)
        if not isinstance(m, int):
            raise TypeError('m (the number of mixture components) must be an integer')
        if m <= 0:
            raise ValueError('m (the number of mixture components) must be positive')
        self.m = m
        
        self.typePrior_mu = typePrior_mu
        self.initialize()
    
    def initialize_sigma(self): 

        self.sigma = empty(shape = (self.m, self.p, self.p))
        self.sigma_clusters()

    def mu_clusters(self):

        self.mu = dot(self.Resp, self.y.T) / self.Resp.sum(axis = 1).reshape(self.m, 1)
 
    def sigma_clusters(self):
        
        for Resp_i, mu, sigma in zip(self.Resp, self.mu, self.sigma):
            y_mu = self.y - mu.reshape(self.p, 1)
            sigma[:] = dot(Resp_i * y_mu, y_mu.T) / Resp_i.sum()


class vq(mog):
    """
    high alfa: small and separated clusters, approaching hard clustering or WTA rule
    alfa = 1: mog
    low alfa: smooth, fuzzy like, large and overlapping clusters
    """
       
    def InferandLearn(self, max_iter_nr = 100, alfa = 1): #entropy_delta = 1e-3 
        
        self.alfa = alfa
        lm.InferandLearn(self, max_iter_nr = max_iter_nr)
    
    def Resppower(self): self.Resp = powerer(self.Resp, self.alfa)
 

class hardvq(vq):
    """alfa->infinite (so hard clsutering or Winner-Take-All rule [WTA])"""
    
    def normalizeResp(self): pass
    
    def Resppower(self):

        indices_max = nanargmax(self.Resp, axis=0)
        self.Resp = zeros((self.m, self.n))
        self.Resp[indices_max, arange(self.n)] = 1.

class hard2vq(hardvq):
    """WTA and clusters equally probable"""

    def initialize_pi(self): self.pi = ones(self.m, dtype = 'float') / self.m
    
    def pi_clusters(self): pass
    

class kmeans(hard2vq):
    """WTA, clusters equally probable and covariance matrices all equal to I"""
    
    def initialize_sigma(self):
        
        self.sigma = zeros(shape = (self.m, self.p, self.p))
        self.sigma[:, self.arangep, self.arangep] = ones(self.p)

    def sigma_clusters(self): pass


class mofa(mixture):
    """
    Mixture of Factor Analyzers
    Based on ref.2, 3, 10 and 11.
    NB Formula (11) in ref.3, concerning the learning of the Uniquenesses
    of each Factor Analyzer, seems to be not correct when such covariance
    matrices have not be fixed equal a priori: in the last case replace
    the N at the denominator with the sum of Responsibilities for each
    Factor Analyzer (FA). 
    """
    
    k = None

    def __init__(self, y, m, k, commonUniq = False, typePrior_mu = 'normal'):
        """
        m: the nr of mixture FA components
        k: the (tuple of) nr of hidden factors for each FA component
        commonUniq: a boolean set to True if we want to learn a common
                    Uniquenesses term for each FA component, as in ref.2
                    (at regard see class method postprocess_R).
        typePrior_mu: set to 'normal' or 'uniform' defining the pdf used
                      to initialize the self.mu terms.
        """

        super(mofa, self).__init__(y)
        if not isinstance(m, int):
            raise TypeError('m (the number of mixture components) must be an integer')
        if m <= 0:
            raise ValueError('m (the number of mixture components) must be positive')
        self.m = m
        
        if hasattr(k, '__iter__'):
            try: map(int, k) 
            except ValueError:
                raise ValueError('Specify integers for the iterable of latent factors nr')
            if len(k) < m:
                raise Exception('Specify as many latent factors as components nr')
            for ki in k:
                if ki <= 0:
                    raise Exception('Specify positive integers as latent factors nr')
        else:
            try: k = int(k)
            except ValueError:
                raise Exception('Specify an integer for the nr of latent factors')
            if k <= 0:
                raise Exception('Specify a positive integer for the nr of latent factors')
            k = [k] * m
        self.k = tuple(k)
        
        self.commonUniq = commonUniq
        self.typePrior_mu = typePrior_mu
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiating the Mixture of Factor Analyzers
        # Giving None in input to fa class instances, because 
        # superclass lm just got data y when given in input to mofa 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.fas = [fa(None, ki) for ki in self.k]
        self.initialize()

    def initialize_sigma(self):

        self.sigma = zeros(shape = (self.m, self.p, self.p))
        for sg, fa in zip(self.sigma, self.fas):
            sg[:] = dot(fa.C, fa.C.T) + diag(fa.R)
    
    def initializelogLH(self):
        
        self.logLH = -inf
        self.deltalogLH = inf
        self.logLH_tracks = []
        self.logLH__break = False
        for fa in self.fas:
            fa.initializelogLH()
            fa.betaInferenceLemma() #betaInference() 
        self.logLH_const = -.5 * self.p * log(2. * pi)

        # An help from already available Resp values
        self.LH_temp = empty((self.m, self.n))    
    
    def mu_clusters(self):
        """
        This function is by-passed because self.mu's are calculated in the
        coupled matrix equations in sigma-clusters, together to fa.C matrix.
        """
        
        pass

    def sigma_clusters(self):

        sumResp_all = self.Resp.sum(axis = 1)

        for i, (k, mu, Resp, sumResp, cls) in enumerate(zip(self.k, \
                                self.mu, self.Resp, sumResp_all, self.fas)):
            
            y = self.y
            p, n = self.p, self.n 
            _mu = mu.reshape(p, 1)
            cls.betaInferenceLemma() #betaInference()
            beta, C, R  = cls.beta, cls.C, cls.R
            
            ######################################
            # E step (the following two rows) 
            ######################################
            Exy = dot(beta, y - _mu)
            ExyBlock = vstack((Exy, ones((1, n))))
            
            RespExy = Resp * Exy
            sumRespExy = RespExy.sum(axis=1).reshape(k, 1)
                
            RespyExy = dot(Resp * y, Exy.T)
            RespyExyBlock = dot(Resp * y, ExyBlock.T)

            RespExxy = dot(RespExy, Exy.T) - sumResp * dot(beta, C)
            RespExxy[arange(k), arange(k)] += sumResp
            RespExxyBlock = vstack((hstack((RespExxy, sumRespExy)), 
                                           append(sumRespExy.T, sumResp)))

            ######################################
            # M step
            ######################################
            try:
                Cmu = dot(RespyExyBlock, inv(RespExxyBlock))
                cls.C[:] = Cmu[:, :-1]
                mu[:] = Cmu[:, -1]
                cls.R[:] = diag(dot(Resp * y, y.T) - \
                                dot(Cmu, dot(Resp * ExyBlock, y.T)))
            except linalg.LinAlgError:
                print 'Mixture Component %d-th disappeared' % i
                self.pi[i] = 0.
        
        self.postprocess_R()
        for sg, fa in zip(self.sigma, self.fas):
            sg[:] = dot(fa.C, fa.C.T) + diag(fa.R)

    def postprocess_R(self):
        """
        This function process all fa.R based on self.commonUniq settings,
        so weighting them by the sum of the component's Responsabilities,
        or taking them equal to a mean term.
        This difference emerges from what reported in the class doc about
        the distinct post-processing versions given in ref.2 and 3.
        """
        
        if not(self.commonUniq):
            for Resp, fa in zip(self.Resp, self.fas):
                fa.R[:] /= Resp.sum()
        else:
            Rmean = zeros(self.p) 
            for fa in self.fas:
                Rmean += fa.R
            Rmean /= self.n
            for fa in self.fas:
                fa.R[:] = Rmean
    
    def logLikelihood_Old(self):
        """TODO: Review this!"""
    
        logLH_old = self.logLH
        LH = 0.
        for fa, pi, mu in zip(self.fas, self.pi, self.mu):
            fa.logLikelihood(mu = mu.reshape(self.p, 1))
            LH += exp(fa.logLH) * pi
        self.logLH = log(LH)
        self.deltalogLH = self.logLH - logLH_old
        self.logLH_tracks.append(self.logLH)
    

class icaMacKay(lm):
    """Based on ref.7, 9."""
    
    def __init__(self, y):
        
        super(icaMacKay, self).__init__(y)
        self.k = self.p
        self.initialize()
        
    def initialize(self):
        
        self.Q = self.R = None
        self.A = random.uniform(-1, 1, size = (self.p, self.p)) #rand(self.p, self.p) #
        self.A_start = empty((self.p, self.p))
        self.A_start[:] = self.A
    
    def nonlinear_map(self, z): return -tanh(z)

    def InferandLearn(self,
                      maxinner_iter_nr = inf,
                      maxouter_iter_nr = 10,
                      eta = 'adaptive',
                      bias_eta = 100.,
                      verbose = False):
        
        _nonlinear_map = self.nonlinear_map
        if eta == 'adaptive':
            eta = 1. / bias_eta
            _switch = 1
        else: eta, _switch = .002, 0
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Iterations start
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        while True:
            for iter_nr in xrange(maxouter_iter_nr):
                for i, x in enumerate(self.y.T):
                    if i == maxinner_iter_nr: break
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Put x through a linear mapping
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    a = dot(self.A, x)
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Put a through a nonlinear map
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    z = _nonlinear_map(a)
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Put a back through A
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    xi = dot(self.A.T, a)
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Adjust the weights in accordance with 
                    # If eta scales adaptively as 1/n, we have to add a term
                    # to n, otherwise the NaN values will appear in the first
                    # iterations of the algorithm.
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    eta = (1 -_switch) * eta + _switch / (i + bias_eta)
                    self.A += eta * (self.A + outer(z, xi))
                if any(isnan(self.A.ravel())):
                    if verbose:
                        print 'Got NaN at iter %d-th! Re-init unmix matrix A...' % (iter_nr + 1)
                    self.initialize()           
            if any(isnan(self.A.ravel())):
                if verbose:
                    print 'Got NaN after %d iterations! Re-start and re-init unmix matrix A...' % self.max_iter_nr
                self.initialize()
                continue

            break
