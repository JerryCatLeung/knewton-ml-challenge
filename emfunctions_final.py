import numpy as np
from warnings import warn

def nan_warning( message_, arr_ ):
    """
    
    nan_warning is used to warn the user if one of the functions taken
    as an input to fmin_ncg returns a nan value.  This iss very helpful
    for debugging.

    """
    if np.any( np.isnan( arr_ ) ):
        warn( message_ )

def log_post( pr_, pi_, alph_, delt_ ):
    """

    log_post calculates log posterior of the finite bernoulli mixture
    IRT model.  Prior on pi_ is uniform (i.e. not explicitly included)
    while priors on alph_ and delt_ are normal, N(0,1) and N(1,2), respectively.
    These priors are used to deal with items with very few observations
    (to keep them from causing numerical problems) and to deal with the fact
    that the likelihood function doesn't have a unique maximum.  

    """

    (N,K) = pr_.shape

    LL_1 = np.empty( [N,1] )
    LL_1.fill( np.nan )

    LL_1 = np.sum( pi_ * pr_, 1 ).reshape( [N,1] )
    LL_1 = np.log( LL_1 )

    LL = -np.sum( LL_1 )

    # add in alph_ prior
    LL += np.float( np.dot( alph_, alph_.T ) / 2. )
    
    # add in delt_ prior
    LL += np.float( np.dot( ( delt_ - 1. ), ( delt_ - 1. ).T ) / 8. ) 

    nan_warning( 'LL is nan', LL )
    return LL

def e_log_post( alph_, thet_, delt_, x_, pi_, fnPr_, gam_ ):
    """
    e_log_pst calculates the expected value of the log
    posterior conditional on the model parameters AND on the 
    latent variables assigning students to ability

    Inputs:
    thet_: (1 x K) matrix of latent abilities
    delt_: (1 x D) matrix of item 'discrimination' parameter
    alph_: (1 x D) matrix of item 'difficulty' parameter 
    x_: (N x D) data matrix
    pi_: (1 x K) matrix of probabilities of being in ability class k
    gam_: (N x K) matrix of responsibilities

    """
    (N,D)  = x_.shape        # get dimensions
    K      = thet_.shape[1]  # get more dimensions

    P      = np.empty( [K,D] )  # initialize P(k,d) probability matrix
    P.fill( np.nan )
    LL_1   = np.empty( [N,K] )  # initialize intermediate matrix 
                                # of elements to sum
    LL_1.fill( np.nan )

    P = fnPr_( thet_, delt_, alph_ ) # (1) Generate P(k,d) matrix with initial
                                     # thet_, delt_ and alph_
    
    # Dealing with zero and one probabilities (cheating a bit)
    # ( Might not be necessary when we introduce priors on alph_ and delt_)
    P = np.maximum( P, np.finfo( np.float64 ).eps )
    P = np.minimum( P, 1 - np.finfo( np.float64 ).eps )

    # (2) LL_1(n,k) = gam_(n,k) * ln( pi_(k) )
    LL_1 = gam_ * np.log( pi_ )

    # (3) LL_1(n,k) = (2) + sum_d( P(k,d) * x_(n,d) )
    LL_1 += np.dot( np.greater( x_, 0 ), np.log( P ).T ) * gam_
    
    # (4) LL_1(n,k) = (3) + sum_d( ( 1 - P(k,d) ) * ( 1 - x_(n,d) ) )
    LL_1 += np.dot( np.greater( 1 - x_, 0 ), np.log( 1 - P ).T ) * gam_

    LL = -np.sum(LL_1)  # (5) LL = sum_n,k( LL_1 )
    
    LL += np.float( np.dot( alph_, alph_.T ) / 2. )

    LL += np.float( np.dot( ( delt_ - 1. ), ( delt_ - 1. ).T ) / 8. )
    nan_warning( 'LL is nan', LL )
    return LL

def e_log_post_alph( alph_, thet_, delt_, x_, pi_, fnPr_, gam_ ):
    """
    
    e_log_post_alph wraps e_log_post so that paramter order 
    matches parameter order of gradient_alph and hessian_alph.
    It also reshapes input since fmin_ncg expects a vector,
    not a multidimensional array.

    """
    D = alph_.shape[0]

    alph_ = alph_.reshape( [1,D] )
    
    LL = e_log_post( alph_, thet_, delt_, x_, pi_, fnPr_, gam_ )

    return LL

def e_log_post_delt( delt_, alph_, thet_, x_, pi_, fnPr_, gam_ ):
    """
    
    e_log_post_alph wraps e_log_post so that paramter order 
    matches parameter order of gradient_delt and hessian_delt.
    It also reshapes input since fmin_ncg expects a vector,
    not a multidimensional array.

    """
    D = delt_.shape[0]

    delt_ = delt_.reshape( [1,D] )

    LL = e_log_post( alph_, thet_, delt_, x_, pi_, fnPr_, gam_ )
    
    return LL

def e_log_post_thet( thet_, alph_, delt_, x_, pi_, fnPr_, gam_ ):
    """
    
    e_log_post_alph wraps e_log_post so that paramter order 
    matches parameter order of gradient_thet and hessian_thet.
    It also reshapes input since fmin_ncg expects a vector,
    not a multidimensional array.

    """
    K = thet_.shape[0]

    thet_ = thet_.reshape( [1,K] )

    LL = e_log_post( alph_, thet_, delt_, x_, pi_, fnPr_, gam_ )
    
    return LL

def update_pi_k( gam_ ):
    
    (N,K) = gam_.shape
    
    pi = np.empty( [1,K] )
    pi.fill( np.nan )

    pi = np.mean( gam_, 0 ).reshape( [1,K] )

    nan_warning( 'pi_k is nan', pi )
    return pi

def prob_kd( thet_, delt_, alph_ ):
    """
    prob_ki calculates a matrix of probabilities of getting question i
    correct condition on being of ability k

    Inputs:
    thet_: (1 x K) matrix of latent abilities
    delt_: (1 x D) matrix of item 'discrimination' parameter
    alph_: (1 x D) matrix of item 'difficulty' parameter 

    Output:
    prob: (N x K) matrix of probabilities
    
    """

    K = thet_.flatten().shape[0]   # cols
    D = delt_.flatten().shape[0]   # rows
 
    prob = np.nan*np.zeros( [K,D] ) # initialize probability matrix

    prob = np.tile( thet_.T, [1,D] )  # (1) prob(k,i) = thet_(k)
    prob = prob - alph_               # (2) prob(k,i) = thet_(k)-alph_(i)
    prob = prob * delt_               # (3) prob(k,i) = delt_(i)(thet_(k)-alph_(i))
    prob = np.exp( prob )             # (4) prob(k,i) = exp{phi(k,i)}
                                      #     where phi(k,i) = (3)
    prob = prob / ( 1 + prob )        # (5) prob(k,i) = (4)/(1-(4))
    

    nan_warning( 'prob_kd matrix has nan values', prob )
    return prob          # return K x D matrix of probabilities

def prob_st( x_, pr_ ):

    (K,D) = pr_.shape
    N     = x_.shape[0]

    pr_st = np.empty( [N,K] )
    pr_st.fill( np.nan )

    for i in np.arange( 0, K ):
        a = np.power( pr_[i,:], np.equal( x_, 1 ) )
        b = np.power( 1 - pr_[i,:], np.equal( x_, 0 ) )
        
        pr_st[:,i] = np.prod( a * b, 1).reshape( [N,] )

    nan_warning( 'pr_st is nan', pr_st )
    return pr_st          # returns N x K of student probabilities

def gamma_nk( pr_, pi_ ):
    """
    gamma_nk calculates a matrix of 'responsibilities' of ability (k)
    for student (n) -- can be interpreted as the posterior probability
    that student n has ability k

    Inputs:
    pr_: (N x K) matrix of student (n) likelihood conditional on ability (k)
    pi_: (1 x K) matrix of probabilities of being in ability class k
    
    Output:
    gamma: (N x K) matrix of repsonsibilities (rows should sum to 1)
    
    """

    (N,K) = pr_.shape         # (cols,rows)

    gamma = np.empty( [N,K] ) # initialize gamma matrix
    gamma.fill( np.nan )

    gamma = pr_ * pi_                    # (1) gamma(n,k) = pi_(k)*pr_(n,k)
    gamma = gamma / np.dot( pr_,pi_.T )  # (2) gamma(n,k) = pi_(k)*pr_(n,k)/
                                         #                  sum_n(pi_k*pr_(n,k)

    nan_warning( 'gam_nk matrix has nan values', gamma )
    return gamma          # return N x K matrix of gammas (responsibilities)

def gradient_delt( delt_, alph_, thet_, x_, pi_, fnPr_, gam_ ):
    """
    gradient_delt returns the delt_ components of the gradient 
    of the posterior expectation of the log likelihood with 
    respect to the latent variable Z

    Inputs:
    thet_: (1 x K) matrix of latent abilities
    delt_: (1 x D) matrix of item 'discrimination' parameter
    alph_: (1 x D) matrix of item 'difficulty' parameter 
    x_: (N x D) data matrix
    pi_: (1 x K) matrix of probabilities of being in ability class k
    fnPr_: function that returns a (K x D) matrix of probabilities (see prob_kd)
    gam_: (N x K) matrix of repsonsibilities

    Output:
    grad: (1 x D) gradient with respect to delt_ component
    
    """ 

    (N,D)  = x_.shape        # get dimensions
    K      = thet_.shape[1]  # get more dimensions
    delt_ = delt_.reshape( [1,D] )

    grad_1 = np.empty( [N,D] )  # initialize intermediate matrix
    grad_1.fill( np.nan )
    P      = np.empty( [K,D] )  # initialize P(k,d) probability matrix
    P.fill( np.nan )

    grad   = np.empty( [1,D] )  # initialize alpha component gradient
    grad.fill( np.nan )

    P = fnPr_( thet_, delt_, alph_ ) # (1) Generate P(k,d) matrix with initial
                                     # thet_, delt_ and alph_

    # (2) grad_1(n,d) = sum_k( gam_(n,k) * P(n,k) * (thet_k - alph_(d)) )
    grad_1 = np.dot( gam_ * thet_, P ) - np.dot( gam_, P * alph_)                 

    # (3) grad_1(n,d) = (2) + sum_k( gam_(n,k) * x_(n,k) * (thet_(k) - alph(d)) )
    grad_1 = ( np.sum( gam_ * thet_, 1).reshape( [N,1] ) * x_ 
               - x_ * alph_ - grad_1 ) 
                                                          
    grad = -np.nansum( grad_1, 0 )    # (4) grad(d) = sum_n( grad_1(n,d) ) 

    grad += delt_.flatten() / 4.      # (5) grad(d) = (4) + delt_(d) / 4.
        
    nan_warning( 'delt_ component of gradient has nan values', grad )
    
    return grad                      # (5) return gradient component

def gradient_alph( alph_, thet_, delt_, x_, pi_, fnPr_, gam_  ):
    """
    gradient_alph returns the alph_ components of the gradient 
    of the posterior expectation of the log likelihood with 
    respect to the latent variable Z

    Inputs:
    thet_: (1 x K) matrix of latent abilities
    delt_: (1 x D) matrix of item 'discrimination' parameter
    alph_: (1 x D) matrix of item 'difficulty' parameter 
    x_: (N x D) data matrix
    pi_: (1 x K) matrix of probabilities of being in ability class k
    fnPr_: function that returns a (K x D) matrix of probabilities (see prob_kd)
    gam_: (N x K) matrix of repsonsibilities

    Output:
    grad: (1 x D) gradient with respect to alph_ component
 
    """
    
    (N,D)  = x_.shape        # get dimensions
    K      = thet_.shape[1]  # get more dimensions
    alph_ = alph_.reshape( [1,D] )

    grad_1 = np.empty( [N,D] )  # initialize intermediate matrix
    grad_1.fill( np.nan )
    P      = np.empty( [K,D] )  # initialize P(k,d) probability matrix
    P.fill( np.nan )

    grad   = np.empty( [1,D] )  # initialize alpha component gradient
    grad.fill( np.nan )

    P = fnPr_( thet_, delt_, alph_ ) # (1) Generate P(k,d) matrix with initial
                                     # thet_, delt_ and alph_

    # (2) grad_1(n,d) = sum_k( gam_(n,k)*P(n,k) * delt_(d) )
    grad_1 =  np.dot( gam_, P * delt_)                 
    
    # (3) grad_1(n,d) = sum_k( gam_(n,k) * x_(n,d) * -delt_(d) ) - (2)
    grad_1 = x_ * -delt_ + grad_1
                                                          
    grad = -np.nansum( grad_1, 0 )    # (4) grad(d) = sum_n( grad_1(n,d) ) 

    grad += alph_.flatten()           # (5) grad(d) = (4) + alph_(d)
    
    nan_warning( 'alph_ component of gradient has nan values', grad )
    return grad                      # (6) return gradient component

def hessian_alph( alph_, thet_, delt_, x_, pi_, fnPr_, gam_  ):
    """
    hess_alph returns the alph_ component hessian 
    of the posterior expectation of the log likelihood with 
    respect to the latent variable Z

    Inputs:
    thet_: (1 x K) matrix of latent abilities
    delt_: (1 x D) matrix of item 'discrimination' parameter
    alph_: (1 x D) matrix of item 'difficulty' parameter 
    x_: (N x D) data matrix
    pi_: (1 x K) matrix of probabilities of being in ability class k
    fnPr_: function that returns a (K x D) matrix of probabilities (see prob_kd)
    gam_: (N x K) matrix of repsonsibilities

    Output:
    hess: (D x D) diagonal hess with respect to alph_ component
 
    """
    
    (N,D)  = x_.shape        # get dimensions
    K      = thet_.shape[1]  # get more dimensions
    alph_ = alph_.reshape( [1,D] )

    hess_1 = np.empty( [N,D] )  # initialize intermediate matrix
    hess_1.fill( np.nan )
    P      = np.empty( [K,D] )  # initialize P(k,d) probability matrix
    P.fill( np.nan )

    hess   = np.zeros( [D,D] )  # initialize alpha component gradient

    P = fnPr_( thet_, delt_, alph_ ) # (1) Generate P(k,d) matrix with initial
                                     # thet_, delt_ and alph_

    # (2) hess_1(n,d) = sum_k( gam_(n,k)*P(k,d)*(1-P(k,d)*delt_(d) )
    hess_1 =  np.dot( gam_, P * ( 1 - P ) * np.square( delt_ ) )
    
    # (3) remove terms from (2) where x_ is missing
    hess_1 *= ~np.isnan( x_ ) 

    # (4) hess(d,d) = diagonalize( sum_n( hess_1(n,d) ) + 1.  )
    hess = np.diagflat( np.nansum( hess_1, 0 ) + np.ones( [D,] ) )   
    
    nan_warning( 'alph_ component of gradient has nan values', hess )
    return hess                      # (5) return component hessian


def hessian_delt( delt_, alph_, thet_, x_, pi_, fnPr_, gam_  ):
    """
    hess_delt returns the delt_ component hessian 
    of the posterior expectation of the log likelihood with 
    respect to the latent variable Z

    Inputs:
    thet_: (1 x K) matrix of latent abilities
    delt_: (1 x D) matrix of item 'discrimination' parameter
    alph_: (1 x D) matrix of item 'difficulty' parameter 
    x_: (N x D) data matrix
    pi_: (1 x K) matrix of probabilities of being in ability class k
    fnPr_: function that returns a (K x D) matrix of probabilities (see prob_kd)
    gam_: (N x K) matrix of repsonsibilities

    Output:
    hess: (D x D) diagonal hessian with respect to delt_ component
 
    """
    
    (N,D)  = x_.shape        # get dimensions
    K      = thet_.shape[1]  # get more dimensions
    delt_ = delt_.reshape( [1,D] )

    hess_1 = np.empty( [N,D] )  # initialize intermediate matrix
    hess_1.fill( np.nan )
    P      = np.empty( [K,D] )  # initialize P(k,d) probability matrix
    P.fill( np.nan )

    hess   = np.zeros( [D,D] )  # initialize delta component hessian

    P = fnPr_( thet_, delt_, alph_ ) # (1) Generate P(k,d) matrix with initial
                                     # thet_, delt_ and alph_

    # (2) hess_1(n,d) = sum_k( gam_(n,k)*P(k,d)*(1-P(k,d)*alph_(d) )
    hess_1 = -2 * np.dot( gam_ * thet_ , P * ( 1 - P ) * alph_ )                 
    
    # (3) hess_1(n,d) = (2) + sum_k( gam_(n,k)^2*P(k,d)*(1-P(k,d))
    hess_1 += np.dot( gam_ * np.square( thet_ ), P * ( 1 - P ) )

    # (3) hess_1(n,d) = (2) + sum_k( alph_(d)^2*P(k,d)*(1-P(k,d))
    hess_1 += np.dot( gam_ , P * ( 1 - P ) * np.square( alph_ ) )

    # (5) remove terms from (4) where x_ is missing
    hess_1 *= ~np.isnan( x_ ) 

    # (6) hess(d,d) = diagonalize( sum_n( hess_1(n,d) ) + 1. / 4. )
    hess = np.diagflat( np.nansum( hess_1, 0 ) + np.ones( [D,] ) / 4. )   
    
    nan_warning( 'delt_ component of gradient has nan values', hess )
    
    return hess                      # (7) return component hessian
