import csv
import numpy as np
from scipy import optimize as opt
import emfunctions_final as em
import datetime as dt

def main():
    # Import data
    
    # open data file
    infile = open('data/astudentData.csv','r')

    # load as object
    csvfile = csv.reader(infile,delimiter=',')

    # pass over labels in first row
    labels = csvfile.next()

    # load (qId,studentId,score) tuples into list
    data = []
    qDict = {}
    qMap  = {}
    sDict = {}
    for line in csvfile:
        data.append((line[0],line[1],line[2]))
        
        try:
            qDict[line[0]]
        except KeyError:
            qMap[len( qDict )] = line[0]
            qDict[line[0]] = len( qDict )

        try:
            sDict[line[1]]
        except KeyError:
            sDict[line[1]] = len( sDict )

    N = len( sDict )
    K = len( qDict )

    x = np.empty( [N,K] )
    x.fill( np.nan )
    for qId, stId, ans in data:
        x[ sDict[stId], qDict[qId] ] = np.int(ans)

    # Initialize parameters and latent variables

    (N,D) = x.shape
    K     = 8                    # number of latent ability classes 

    alpha = np.zeros( [1,D] )    # at prior center
    theta = np.array( [[ -2., -1.5, -1., -.5, .5, 1., 1.5, 2.]] ) 
    delta = np.ones( [1,D] )      # at prior center
    pi    = np.ones( [1,K] ) / K # uniform
    gamma = np.empty( [N,K] )

    # Set tolerance and iteration count
    i     = 0
    crit  = 1e-5
    
    # Open results files

    f = ( open( 'results/emMartianOutput' + 
                dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') 
                + '.txt', 'w' ) )

    fa = ( open( 'results/emMartianAlpha' + 
                dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') 
                + '.txt', 'w' ) )

    fd = ( open( 'results/emMartianDelta' + 
                dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') 
                + '.txt', 'w' ) )

    fp = ( open( 'results/emMartianPi' + 
                dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') 
                + '.txt', 'w' ) )

    fRemove = ( open( 'results/emMartianRemove' + 
                dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') 
                + '.txt', 'w' ) )
    
    # Evaluate initial log posterior
    pr         = em.prob_kd( theta, delta, alpha )
    pr_student = em.prob_st( x, pr ) 
    LL0 = em.log_post( pr_student, pi, alpha, delta ) 
    print 'initial log posterior: {0: f}'.format( LL0 )
 
    LL  = LL0
    prevLL = LL + 1   
    aDiff = 1
    dDiff = 1

    # loop until change in parameters or likelihood is below tolerance (crit)
    while  prevLL - LL > crit and aDiff > crit and dDiff > crit:
        
        # save previous iteration estimates
        prevLL = LL
        prevAlph = alpha
        prevDelt = delta

        # E-step
         
        gamma = em.gamma_nk( pr_student, pi )
        
        # CM-step
        
        # delta component

        delta = ( opt.fmin_ncg(em.e_log_post_delt, delta, em.gradient_delt, disp = 1,
                                    fhess = em.hessian_delt, args = ( alpha, theta, x, pi, em.prob_kd, gamma ) ) )
        delta.tofile(fd,sep=',')
        fd.write('\n')
        delta.resize( [1,D] )

        # alpha component

        alpha = ( opt.fmin_ncg(em.e_log_post_alph, alpha, em.gradient_alph, disp = 1,
                                    fhess = em.hessian_alph, args = ( theta, delta, x, pi, em.prob_kd, gamma ) ) )
        alpha.tofile(fa,sep=',')
        fa.write('\n')
        alpha.resize( [1,D] )
        
        # pi component

        pi = em.update_pi_k( gamma )

        pi.flatten().tofile(fp,sep=',')
        fp.write('\n')

        # Evaluate updated log posterior
        pr         = em.prob_kd( theta, delta, alpha )
        pr_student = em.prob_st( x, pr ) 
        LL = em.log_post( pr_student, pi, alpha, delta)       
    
        # Check max change in parameter values
        aDiff = np.max( np.abs( alpha - prevAlph ) )
        dDiff = np.max( np.abs( delta - prevDelt ) )
        
        print '-------------------------------------------------'
        print 'Max change in alpha: {0:f}'.format( aDiff )
        print 'Max change in delta: {0:f}'.format( dDiff )
        print '-------------------------------------------------'

        print 'log posterior at iteration {0:d}: {1:f}'.format( i+1,LL )
        print '-------------------------------------------------'
        f.write( '{0: f} \n'.format( LL ) )

        i += 1

    # Close results files
    f.close()
    fa.close()
    fd.close()

    print 'FINISHED... ITERATIONS: {0: d}... LOG POSTERIOR: {1: f}'.format( i, LL )

    # Write file of questions to remove
    ind = np.nonzero( delta < 1 )[1]
    
    fRemove.write('\n'.join( qMap[i] for i in ind ) )
    fRemove.close()



if __name__ == '__main__':
    main()
