try:
    import psyco
    psyco.full()
except ImportError:
    pass

try:
    import numpy
    _array  = numpy.array
    _inv    = numpy.linalg.inv
    _dot    = numpy.dot
    _inner  = numpy.inner
    _outer  = numpy.outer
    _vdot   = numpy.vdot
    _tensordot = numpy.tensordot
    _det    = numpy.linalg.det
    _cov    = numpy.cov
    _diag   = numpy.diag
    _ones   = numpy.ones
    _eye    = numpy.eye
    _norma  = numpy.linalg.norm
    _arange = numpy.arange
    _zeros  = numpy.zeros
    _solve  = numpy.linalg.solve
    _Cholesky = numpy.linalg.cholesky
    _argmax = numpy.argmax
    _nanargmax = numpy.nanargmax
    _mean   = numpy.mean
    _average = numpy.average
    _std    = numpy.std
    _mul    = numpy.multiply
    _sum    = numpy.sum
    _prod   = numpy.product
    _sqrt   = numpy.sqrt
    _log    = numpy.log
    _abs    = numpy.abs
    _exp    = numpy.exp
    _pow    = numpy.power
    _hstack = numpy.hstack
    _vstack = numpy.vstack
    _append = numpy.append
except ImportError:
    raise ImportError, "Impossible to import numpy/scipy module"


__name__ = "glgm"
__doc__ = """Notation and references are reported in README file"""



class lm(object):
    """See details explained above"""

    __slots__ = ['y', 'n', 'p']
    y, n, p = _default_values_ = None, 0, None
    cumulate = True
    
    def __new__(cls, y, *args, **kwargs):
        """
        NB It seems to be a bug that python.exe (x86 binaries on wind)
           gives a Deprecation Warning when compiling this code,
           saying that __new__() can't accept arguments
           TODO: every sort of BROADCASTING!!!
        """

        obj = object.__new__(cls, y, *args, **kwargs)
        if hasattr(y, '__iter__'):
            lm.cumulate = kwargs.get('cumulate', True)
            if lm.y is None or not(lm.cumulate): lm.y = numpy.array(y, dtype = 'float32')
            else: lm.y = numpy.concatenate((lm.y, y), axis = 1) #append(lm.y, y, axis = 1) #hstack((lm.y, y))
            lm.p, lm.n = lm.y.shape
        obj.y, obj.n, obj.p, obj.cumulate = lm.y, lm.n, lm.p, lm.cumulate
        return obj

    def mu_y(cls, y = None):
        
        if y is None: y = cls.y
        return numpy.mean(y, axis = 1).reshape(cls.p, 1)
    
    def cov_obs(cls, y = None, cov_bias = 1):
        
        if y is None: y = cls.y
        return numpy.cov(y, bias = cov_bias)
    
    def centered_input(cls): return cls.y - cls.mu_y()

    def erase(cls): cls.y, cls.n, cls.p = lm._default_values_

    def __call__(self, **kw): return self.InferandLearn(**kw)

    def InferandLearn(self, max_iter_nr = 30, **kw):
        """
        Inference and Learning method to be delegated by subclasses.
        Here EM algorithm's iterations will start.
        """

        E_EM, M_EM  = self.Inference, self.Learning
        logLH, Break = self.logLikelihood, self.break_condition
        self.logLH_delta = kw.get('logLH_delta', None)
        for kw, val in kw.iteritems(): setattr(self, 'kw', val)
        
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
        
        self.trained = True

    def next(self, **kwargs):
        """Calling InferandLearn method to run only one iteration"""

        self.InferandLearn(max_iter_nr = 1, **kwargs)
    
    def Inference(self):
        """The E(xpectation) step of EM algorithm"""
        
        pass

    def Learning(self):
        """The M(aximization) step of EM algorithm"""

        pass
    
    def break_condition(self): 
        """A method verifying an ad-hoc condition to exit from EM iter."""

        pass
    
    def logLikelihood(self):
        """The logLikelihood method for the given model"""

        pass  
    


class fa(lm):
    """
    GLGM static data modeling in fa, ppca and pca
    Based on ref.1
    Model:
        A = 0
        x = x0 = w0,        w0 = N(0, Q)
        y = y0 = Cx + v0,   v0 = N(0, R)
    From which
        y ~ N(0, CQCT + R) 
    and in order to solve any model degeneracy
        Q = I
        R is diagonal
    then
        y   ~ N(0, C*CT + R)
        x_y ~ N(beta*y, I-beta*C)   useful for the Inference task
        beta = CT(C*CT + R)^-1
        CT = C.T
    Here C is also called the factor loading matrix,
    the diagonal elements of R as the uniquenesses,
    and v the sensor noise.
    This is a static glgm => A = None | []
    Hint: apply Template Design Pattern to scale-down from fa
          to spca (or ppca), pca and whitening sub-classes.
    """

    k = None
    
    def __init__(self, y, k):
        
        try: k = int(k)
        except: raise Exception, "Specify a valid positive integer for the nr latent factors"
        if k <= 0: raise Exception, "Specify a positive integer for the nr latent factors"
        if k > self.p: raise Exception, "Specify a number of latent factors lower or equal than observables variables number"
        setattr(self, 'k', k)

        self.initialize()

    def initialize_Q(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Covariance matrix Q of hidden factors = I matrix
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.Q = _eye(self.k)
    
    def initialize_C(self):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Following MDP init settings
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        scale = _prod(self.yyT_diag) ** (1./self.p)
        self.C = numpy.random.normal(0, numpy.sqrt(scale / self.k), size = (self.p, self.k))
    
    def initialize_R(self, with_WN = False):
        """There are other (here commented) ways to init R"""
        
        self.R = self.yyT_diag
        if with_WN: self.R += numpy.random.randn(self.p)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # From pag.531 of ref.4 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ## self.R = (1-.5*self.p/self.k)*self.yyT_diag

    def initialize_logLH(self):
        
        self.logLH = -numpy.inf
        self.delta_logLH = numpy.inf
        self.logLH_const = -.5 * self.p * _log(2. * numpy.pi)
        self.logLH_tracks = []
        self.logLH__break = False

    def initialize(self):
        """Initialization step: C and other vars, get observed data covariance"""

        self.arangek, self.arangep = _arange(self.k), _arange(self.p)
        
        self.yyT = self.cov_obs()
        self.yyT_diag = self.yyT[self.arangep, self.arangep]
        self.initialize_C()
        self.initialize_R()
        self.initialize_Q()
        self.initialize_logLH()

    def InferandLearn(self, max_iter_nr = 20, logLH_delta = 1e-3,
                      inferwithLemma = True, **kwargs):

        self.betaInferenceMethod = self.betaInferenceLemma \
                            if inferwithLemma else self.betaInference

        lm.InferandLearn(self, max_iter_nr = max_iter_nr, logLH_delta = logLH_delta)
        
    def break_condition(self):
        
        if -self.logLH_delta < self.delta_logLH < self.logLH_delta:
            self.logLH__break = True; return True
        return False
    
    def betaInferenceLemma(self):

        C, CT, R_inv = self.C, self.C.T, self.R ** -1
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Applying the matrix inversion lemma
        # See: - pag.334 in ref.1
        #      - aka Sherman-Morrison-Woodbury formula at pag.50 
        #        in ref.8 (formula (2.1.4)
        #      - aka binomial inverse theorem at 
        #        http://en.wikipedia.org/wiki/Binomial_inverse_theorem
        #      - or derived from matrix blockwise inversion as in 
        #        http://en.wikipedia.org/wiki/Invertible_matrix#Blockwise_inversion
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        beta_temp = _mul(CT, R_inv)
        beta = _dot(beta_temp, C)
        beta[self.arangek, self.arangek] += 1.
        beta = -_dot(C, _dot(_inv(beta), beta_temp))
        beta[self.arangep, self.arangep] += 1.
        self.logLH_temp = _mul(R_inv.reshape(self.p, 1), beta) ## R_inv * beta #_mul(R_inv, beta)
        self.beta = _dot(beta_temp, beta)

    def betaInference(self):

        C, R = self.C, self.R
        CT = C.T
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Applying the classical method to invert beta
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        beta = _dot(C, CT)
        beta[self.arangep, self.arangep] += R
        self.logLH_temp = beta # = _inv(beta)
        self.beta = _dot(CT, _inv(beta)) #beta)

    def Inference(self):
        """
        E step of EM algorithm
        Inference of sufficient statistic E(x|y) (here x_y)
        NB I tried to compute beta via the matrix inversion lemma
           but performances doesn't seem to be better than applying the ordinary formula.
        NB beta.ravel()[_arange(0, k**2, k) + _arange(k)] += 1.   #<--- code at left is inefficient!
           beta.ravel()[_arange(0, p**2, p) + _arange(p)] += 1.   #<--- code at left is inefficient!
           self.V.ravel()[_arange(0, k**2, k) + _arange(k)] += 1. #<--- code at left is inefficient!
        TODO Test if betaInferenceLemma is advantageous at high p values.
        """
        
        self.betaInferenceMethod()
        self.V = - _dot(self.beta, self.C)
        self.V[self.arangek, self.arangek] += 1.

    def Learning_R(self):
        """
        Learning R in M step of EM algorithm
        It is necessary to learn R separately in order to give
        sub-classes able to override the present method
        """
        
        self.R = self.yyT_diag - _sum(_mul(self.C, self.delta), axis=1) / self.n

    def Learning(self):
        """
        M step of EM algorithm
        Computing delta and gamma (ref.1 at pag.335).
        Learning and updating model's parameters C and R
        """
        
        delta = _dot(self.yyT, self.beta.T)
        self.gamma = self.n * (_dot(self.beta, delta) + self.V)
        delta *= self.n
        self.delta = delta
        self.C = _dot(self.delta, _inv(self.gamma))
        self.Learning_R()

    def logLikelihood(self, mu = None):
        """
        Log-likelihood (LogLH)
        This the LogLH per sample, apart the constant self.logLH_const and
        factor .5 both affecting very sligthly the logLH values convergence
        to a (local, hopefully global) maximum.
        """
        
        logLH_old = self.logLH
        _yyT =  self.yyT - _dot(mu, mu.T) if mu != None else self.yyT 
        self.logLH = self.logLH_const - .5 * (-_log(_det(self.logLH_temp)) + \
                     _vdot(_yyT, self.logLH_temp))
        self.delta_logLH = self.logLH - logLH_old
        self.logLH_tracks.append(self.logLH)

    def infer(self):
        
        if self.trained: return self.get_latent(), self.infer_observed()
        print 'Not even trained!'; return

    def get_expected_latent(self):
        
        if not(self.trained): self.InferandLearn()
        return numpy.dot(self.beta, self.y)

    def infer_observed(self, noised = False):
        
        if not(self.trained): self.InferandLearn()
        inf_y = numpy.dot(self.C, self.get_expected_latent()) + self.mu_y()
        if noised: return inf_y + numpy.random.multivariate_normal(numpy.zeros(self.p), numpy.diag(self.R), self.n).T
        return inf_y
    
    def get_new_observed(self, input, noised = False, centered = False):
        """
        input is: the nr. of OR the latent samples (they'd be WN, but actually I don't check)
                  for which to return new observations based on inference results 
        """

        if not(self.trained): self.InferandLearn()
        if isinstance(input, int): input = numpy.random.normal(size = (self.k, input))
        new_obs = _dot(self.C, input) + self.mu_y()
        if centered: new_obs -= self.mu_y(new_obs)
        if noised:
            new_obs += numpy.random.multivariate_normal(_zeros(self.p),
                                                        _diag(self.R), input.shape[1]).T
        return new_obs


class spca(fa):

    def initialize_R(self):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Covariance matrix R of observations = const*I matrix
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.R = _ones(self.p) * _mean(self.yyT_diag)

    def Learning_R(self):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Covariance matrix R of observations = const*I matrix
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.R = _ones(self.p) * _mean(self.yyT_diag - _sum(_mul(self.C, self.delta), axis=1) / self.n)


class ppca(spca): pass


class pca(fa):
    """
    See ref. 1 and 15
    TODO: Check EM method and in particular how to get an orthonormal basis,
          in order to obtain results comparable to those from the svd approach
    """
    
    def __init__(self, y, k = None):
        
        if k is None: setattr(self, 'k', self.p)
        else:
            try: k = int(k)
            except: raise Exception, "Specify a valid positive integer for the nr latent factors"
            if k <= 0: raise Exception, "Specify a positive integer for the nr latent factors"
            if k > self.p:
                print 'The number of latent factors should be <= observables variables number.'
                print 'Now it has been automatically set equal to the observables variables number'
                setattr(self, 'k', self.p)
            else: setattr(self, 'k', k)

        self.initialize()
    
    def initialize_C(self):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Following MDP init settings
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        scale = 1. #_prod(self.yyT_diag) ** (1./self.p)
        self.C = numpy.random.normal(0, numpy.sqrt(scale / self.k), size = (self.p, self.k))
        self.C = numpy.array([e / _norma(e) for e in self.C.T]).T
    
    def initialize_R(self):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Covariance matrix R of observations = Zeros matrix
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.R = _zeros(self.p)

    def Inference(self): self.betaInference()
    
    def betaInference(self):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Read note 10, pag.318, ref.1 about the fomula of beta
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        CT = self.C.T
        self.beta = _dot(_inv(_dot(CT, self.C)), CT)

    def InferandLearn(self, max_iter_nr = 100, svd_on = True, **kwargs):
        
        self.V = _zeros((self.k, self.k))
        
        if svd_on:
            setattr(self, 'svd_on', True)
            self.svd()
        else:
            setattr(self, 'svd_on', False) 
            lm.InferandLearn(self, max_iter_nr = max_iter_nr, **kwargs)
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # According to ref.15: "...The columns of C will span the
            # space of the first k principal components. (To compute
            # the corresponding eigenvectors and eigenvalues explicitly,
            # the data can be projected into this-dimensional subspace 
            # and an ordered orthogonal basis for the covariance in the
            # subspace can be constructed.)..."
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.getOrthogonalBasis()
            
    def getOrthogonalBasis(self):

        try:
            import scipy.linalg as scilinalg
            self.C = scilinalg.orth(self.C)
            #print 'orth(C)=', C_orth
            #u, varsvd, v = numpy.linalg.svd(_cov(_dot(C_orth.T, self.centered_input())), full_matrices = False)
            #self.C = _dot(self.C, v)
        except: print "An error occurred: none orthonormal basis found!"

    def Learning(self):
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # M step of EM algorithm
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.delta = _dot(self.yyT, self.beta.T)
        self.gamma = _dot(self.beta, self.delta)
        self.C = _dot(self.delta, _inv(self.gamma))
    
    def logLikelihood(self): pass
    
    def Learning_R(self): pass

    def logLikelihood(self): pass
    
    def break_condition(self): pass #return self.lse() #based on Least Squares projection error
    
    def lse(self): pass

    def svd(self):
        
        print 'Performing SVD...'
        U, self.VarSvd, V = numpy.linalg.svd(self.yyT, full_matrices = False)
        self.C = U[:, :self.k]
        self.trained = True
    
    def get_expected_latent(self): return _dot(self.beta, self.y)
    
    def get_latent(self): return _dot(self.C.T, self.centered_input())


class whitening(pca):

    def InferandLearn(self, **kwargs):
        
        pca.InferandLearn(self, svd_on = True, **kwargs)
        self.C /= _sqrt(self.VarSvd[:self.k]).reshape(1, self.k)



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
    
        self.arangep = _arange(self.p)
        self.yyT = self.cov_obs()
        self.yyT_diag = self.yyT[self.arangep, self.arangep]
        
        self.initialize_pi()
        self.initialize_mu()
        self.initialize_Resp()
        self.initialize_sigma()
        self.initialize_logLH()

    def initialize_Resp(self):

        self.Resp = numpy.random.rand(self.m, self.n)
        self.normalizeResp()

    def Resppower(self): pass

    def normalizeResp(self): self.Resp /= self.Resp.sum(axis = 0)
    
    def initialize_pi(self):

        self.pi = numpy.random.rand(self.m)
        self.pi /= self.pi.sum()
    
    def pi_clusters(self): self.pi = self.Resp.sum(axis = 1) / self.n

    def Normalprior_mu(self, scale = 1):
        """Normal prior on cluster's centroids"""

        self.mu = numpy.random.multivariate_normal(self.mu_y().ravel(), \
                                self.cov_obs() / scale, self.m).reshape(self.m, self.p)

    def Uniformprior_mu(self):
        """Uniform prior on cluster's centroids"""
        
        _uniform = lambda i, j: numpy.random.uniform(i, j, self.m) 
        self.mu = numpy.array(map(_uniform, numpy.amin(self.y, axis=1), \
                                numpy.amax(self.y, axis=1))).reshape(self.m, self.p)

    def initialize_mu(self):
        """Following a Strategy DP"""
    
        if self.typePrior_mu == 'normal' or not(self.typePrior_mu): self.Normalprior_mu()
        elif self.typePrior_mu == 'uniform': self.Uniformprior_mu()
        
    def initialize_logLH(self):
        
        self.logLH = -numpy.inf
        self.delta_logLH = numpy.inf
        self.logLH_tracks = []
        self.logLH__break = False
        self.logLH_const = -.5 * self.p * _log(2. * numpy.pi)

        # An help from already available Resp values
        self.LH_temp = numpy.empty((self.m, self.n))    
    
    def CrossProdFactory(self, inv_sigma):
        """Curried method in order to speed up the processing"""

        def CrossProdinner(z): return _dot(_dot(z, inv_sigma), z.T)
        return CrossProdinner
    
    def Inference(self):
        """E step of EM algorithm"""

        for Resp_i, pi, mu, sigma, LH in zip(self.Resp, self.pi, self.mu, self.sigma, self.LH_temp):
            yCent_i = self.y - mu.reshape(self.p, 1)
            try:
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # There's no need to multiply const_i by (2*numpy.pi)**self.p
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                const_i = _abs(_det(sigma)) ** -.5
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Currying CrossProdFactory with _inv(self.sigma[i])
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                _CrossProd = self.CrossProdFactory(_inv(sigma)) 
                Resp_i[:] = LH[:] = pi * const_i * _exp(-.5 * _array(map(_CrossProd, yCent_i.T)))
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # If no data in the i-th cluster, fixing pi[i] to zero 
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            except numpy.linalg.linalg.LinAlgError: pi = 0.; pass

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
            const_i = const * _abs(_det(sigma)) ** -.5
            _inv_sigma = _inv(sigma)
            _mu = mu.reshape(self.p, 1)
            LHComp.append(pi * _vdot(self.yyT - _dot(_mu, _mu.T), .5 * _inv_sigma))
        self.logLH = sum(LHComp)
        """
        self.logLH = self.logLH_const + sum(map(_log, self.LH_temp.sum(axis = 0))) / self.n

        self.delta_logLH = self.logLH - logLH_old
        self.logLH_tracks.append(self.logLH)

    def MAP(self): return _argmax(self.Resp, axis = 0)

    def GetExpectedLatent(self): return self.MAP()
    
    # No (WN) noise added
    def InferObs(self): mp = self.MAP(); return [self.y[:, mp == i] for i in xrange(self.m)]

    # No (WN) noise added
    def GetNewObs(self, centered = False): pass    
    
    def GetCompProb(self, obs):
        
        CompProb = []
        for pi, mu, sigma in zip(self.pi, self.mu, self.sigma):
            try:
                const_i = ((2*numpy.pi) ** self.p) * (_abs(_det(sigma)) ** -.5)
                _CrossProd = self.CrossProdFactory(_inv(sigma))
                CompProb.append(pi * const_i * _CrossProd(obs - mu.reshape(self.p, 1)))
            except numpy.linalg.linalg.LinAlgError: CompProb.append(0.)
        return CompProb 

    def entropy(self): pass



class mog(mixture):
    """Based mainly on ref.3, and also on ref.1, 6 and 7"""

    def __init__(self, y, m, typePrior_mu = ''):

        try: m = int(m)
        except: raise Exception, "Specify an integer for the nr latent factors"
        if m <= 0: raise Exception, "Specify a positive integer for the nr latent factors"
        setattr(self, 'm', m)
        
        setattr(self, 'typePrior_mu', typePrior_mu)
        self.initialize()
    
    def initialize_sigma(self): 

        self.sigma = numpy.empty(shape = (self.m, self.p, self.p))
        self.sigma_clusters()

    def mu_clusters(self):

        self.mu = _dot(self.Resp, self.y.T) / self.Resp.sum(axis = 1).reshape(self.m, 1)
 
    def sigma_clusters(self):
        
        for Resp_i, mu, sigma in zip(self.Resp, self.mu, self.sigma):
            y_mu = self.y - mu.reshape(self.p, 1)
            sigma[:] = _dot(Resp_i * y_mu, y_mu.T) / Resp_i.sum()



class vq(mog):
    """
    high alfa: small and separated clusters, approaching hard clustering or WTA rule
    alfa = 1: mog
    low alfa: smooth, fuzzy like, large and overlapping clusters
    """

        
    def InferandLearn(self, max_iter_nr = 100, alfa = 1): #entropy_delta = 1e-3 
        
        self.alfa = alfa
        lm.InferandLearn(self, max_iter_nr = max_iter_nr)
    
    def Resppower(self): self.Resp = _power(self.Resp, self.alfa)
 

class hardvq(vq):
    """alfa->infinite (so hard clsutering or Winner-Take-All rule [WTA])"""
    
    def normalizeResp(self): pass
    
    def Resppower(self):

        indices_max = _nanargmax(self.Resp, axis=0)
        self.Resp = _zeros((self.m, self.n))
        self.Resp[indices_max, _arange(self.n)] = 1.

class hard2vq(hardvq):
    """WTA and clusters equally probable"""

    def initialize_pi(self): self.pi = _ones(self.m, dtype = 'float') / self.m
    
    def pi_clusters(self): pass
    

class kmeans(hard2vq):
    """WTA, clusters equally probable and covariance matrices all equal to I"""
    
    def initialize_sigma(self):
        
        self.sigma = numpy.zeros(shape = (self.m, self.p, self.p))
        self.sigma[:, self.arangep, self.arangep] = _ones(self.p)

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

        try: m = int(m)
        except: raise Exception, "Specify an integer for the nr of mixture components"
        if m <= 0: raise Exception, "Specify a positive integer for the nr of mixture components"
        setattr(self, 'm', m)
        
        if hasattr(k, '__iter__'):
            try: map(int, k) 
            except: raise Exception, "Specify integers for the iterable of latent factors nr"
            if len(k) < m: raise Exception, "Specify as many latent factors as components nr"
            for ki in k:
                if ki <= 0: raise Exception, "Specify positive integers as latent factors nr"
        else:
            try: k = int(k)
            except: raise Exception, "Specify an integer for the nr of latent factors"
            if k <= 0: raise Exception, "Specify a positive integer for the nr of latent factors"
            k = [k] * m
        setattr(self, 'k', tuple(k))
        
        setattr(self, 'commonUniq', commonUniq)
        setattr(self, 'typePrior_mu', typePrior_mu)
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiating the Mixture of Factor Analyzers
        # Giving None in input to fa class instances, because 
        # superclass lm just got data y when given in input to mofa 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.fas = [fa(None, ki) for ki in self.k]
        self.initialize()

    def initialize_sigma(self):

        self.sigma = _zeros(shape = (self.m, self.p, self.p))
        for sg, fa in zip(self.sigma, self.fas): sg[:] = _dot(fa.C, fa.C.T) + _diag(fa.R)
    
    def initialize_logLH(self):
        
        self.logLH = -numpy.inf
        self.delta_logLH = numpy.inf
        self.logLH_tracks = []
        self.logLH__break = False
        for fa in self.fas:
            fa.initialize_logLH()
            fa.betaInferenceLemma() #betaInference() 
        self.logLH_const = -.5 * self.p * _log(2. * numpy.pi)

        # An help from already available Resp values
        self.LH_temp = numpy.empty((self.m, self.n))    
    
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
            Exy = _dot(beta, y - _mu)
            ExyBlock = numpy.vstack((Exy, _ones((1, n))))
            
            RespExy = Resp * Exy
            sumRespExy = RespExy.sum(axis=1).reshape(k, 1)
                
            RespyExy = _dot(Resp * y, Exy.T)
            RespyExyBlock = _dot(Resp * y, ExyBlock.T)

            RespExxy = _dot(RespExy, Exy.T) - sumResp * _dot(beta, C)
            RespExxy[_arange(k), _arange(k)] += sumResp
            RespExxyBlock = numpy.vstack((numpy.hstack((RespExxy, sumRespExy)), 
                                          numpy.append(sumRespExy.T, sumResp)))

            ######################################
            # M step
            ######################################
            try:
                Cmu = _dot(RespyExyBlock, _inv(RespExxyBlock))
                cls.C[:] = Cmu[:, :-1]
                mu[:] = Cmu[:, -1]
                cls.R[:] = _diag(_dot(Resp * y, y.T) - _dot(Cmu, _dot(Resp * ExyBlock, y.T)))
            except numpy.linalg.linalg.LinAlgError:
                print 'Mixture Component %d-th disappeared' % i
                self.pi[i] = 0.
        
        self.postprocess_R()
        for sg, fa in zip(self.sigma, self.fas): sg[:] = _dot(fa.C, fa.C.T) + _diag(fa.R)

    def postprocess_R(self):
        """
        This function process all fa.R based on self.commonUniq settings,
        so weighting them by the sum of the component's Responsabilities,
        or taking them equal to a mean term.
        This difference emerges from what reported in the class doc about
        the distinct post-processing versions given in ref.2 and 3.
        """
        
        if not(self.commonUniq):
            for Resp, fa in zip(self.Resp, self.fas): fa.R[:] /= Resp.sum()
        else:
            R_mean = _zeros(self.p) 
            for fa in self.fas: R_mean += fa.R
            R_mean /= self.n
            for fa in self.fas: fa.R[:] = R_mean
    
    def logLikelihood_Old(self):
        """TODO: Review this!"""
    
        logLH_old = self.logLH
        LH = 0.
        for fa, pi, mu in zip(self.fas, self.pi, self.mu):
            fa.logLikelihood(mu = mu.reshape(self.p, 1))
            LH += _exp(fa.logLH) * pi
        self.logLH = _log(LH)
        self.delta_logLH = self.logLH - logLH_old
        self.logLH_tracks.append(self.logLH)
    


class icaMacKay(lm):
    """Based on ref.7, 9."""
    
    def __init__(self, y):
        
        self.k = self.p
        self.initialize()
        
    def initialize(self):
        
        self.Q = self.R = None
        self.A = numpy.random.uniform(-1, 1, size = (self.p, self.p)) #numpy.random.rand(self.p, self.p) #
        self.A_start = numpy.empty((self.p, self.p)); self.A_start[:] = self.A
    
    def nonlinear_map(self, z): return -numpy.tanh(z)

    def InferandLearn(self,
                      max_inner_iter_nr = numpy.inf,
                      max_outer_iter_nr = 10,
                      eta = 'adaptive',
                      bias_eta = 100.):
        
        _nonlinear_map = self.nonlinear_map
        if eta == 'adaptive': eta = 1. / bias_eta; _switch = 1
        else: eta = .002; _switch = 0
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Iterations start
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        while True:
            for iter_nr in xrange(max_outer_iter_nr):
                for i, x in enumerate(self.y.T):
                    if i == max_inner_iter_nr: break
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Put x through a linear mapping
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    a = _dot(self.A, x)
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Put a through a nonlinear map
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    z = _nonlinear_map(a)
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Put a back through A
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    xi = _dot(self.A.T, a)
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Adjust the weights in accordance with 
                    # If eta scales adaptively as 1/n, we have to add a term
                    # to n, otherwise the NaN values will appear in the first
                    # iterations of the algorithm.
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    eta = (1 -_switch) * eta + _switch / (i + bias_eta)
                    self.A += eta * (self.A + _outer(z, xi))
                if numpy.any(numpy.isnan(self.A.ravel())):
                    if verbose: print 'Got NaN at iter %d-th! Re-init unmix matrix A...' % (iter_nr + 1)
                    self.initialize()           
            if numpy.any(numpy.isnan(self.A.ravel())):
                if verbose:
                    print 'Got NaN after %d iterations! Re-start and re-init unmix matrix A...' % max_iter_nr
                self.initialize()
                continue

            break