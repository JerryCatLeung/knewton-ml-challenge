import csv
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from collections import defaultdict
import warnings

# ignore warnings thrown by natural logarithm function
# these are nan values that arise when there are two few 
# student observations for a question -- this results in them
# not being plotted

warnings.simplefilter( 'ignore', RuntimeWarning )

# open data file
infile = open('data/astudentData.csv','r')

# load as object
csvfile = csv.reader(infile,delimiter=',')

# pass over labels in first row
labels = csvfile.next()

# Sort scores into dictionaries
# with qId and studentId as keys (makes it
# easier to keep track of estimates for 
# Pr[Correct] for students (ability) and
# for questions (difficulty)

qDict = defaultdict(list)
sDict = defaultdict(list)

data = []
for r in csvfile:
    try:
        qDict[r[0]].append([r[1],r[2]])
    except KeyError:
        qDict[r[0]] = ([r[1],r[2]])
        
    try:
        sDict[r[1]].append([r[0],r[2]])
    except KeyError:
        sDict[r[1]] = ([r[0],r[2]])


# Helper function for calculating estimates of
# E[1(Correct)] = Pr[Correct] by student and by question
listMean = lambda x: np.mean(np.array(x,dtype=np.float))

# Means by key over value lists
# Input: dict
# Output: dict (meaned)
def dictMean(dIn):
    dOut = {}
    for k in dIn:
        f = lambda x: x[1]
        tmp = map(f,dIn[k])
        dOut[k] = listMean(tmp)
    return dOut

qP = dictMean(qDict)    # Pr[Correct] by question
sP = dictMean(sDict)    # Pr[Correct] by student


# Helper functions
elem0 = lambda x: x[0]
elem1 = lambda x: x[1]
    
# initialize list
oddsList = []

# initialize counter
i = 0

# Determine median student "ability"
mid = np.median(sP.values())

# By question, get estimate of
# Pr[Correct | ans. q_i correctly] and
# Pr[Correct | ans. q_i incorrectly]
for q in qDict:
    tmpS0 = []
    tmpS1 = []
    if len(qDict[q]):
        for s in qDict[q]: 
            if sP[s[0]] <= mid:
                tmpS0.append(s[1])
            elif sP[s[0]] > mid:
                tmpS1.append(s[1])
        if len(tmpS0) == 0:
            tmpS0 = np.nan
        if len(tmpS1) == 0:
            tmpS1 = np.nan
        oddsList.append((listMean(tmpS0),listMean(tmpS1),q))

# function to calculate log-odds		
fnLogOdds = lambda x: (np.log(x[1]) - np.log(1.-x[1]) - np.log(x[0]) + np.log(1.-x[0]),x[2])  
logOdds = np.array(map(fnLogOdds,oddsList),dtype = [('lO',float),('q',int)])
plt.figure(figsize = (5,4))
plt.plot(map(elem0,oddsList),map(elem1,oddsList),'b.',alpha=0.8)

# Function returns log-odds contours
f = lambda x,k: np.exp(k)*x/(1-x+np.exp(k)*x)
x=np.arange(0.,1.01,.01)

# Plot log-odds contours
for k in [0.,0.5,1.,1.5,2.,2.5,3.]:
	y = f(x,k)
	plt.plot(x,y,'r--',alpha=0.5)

plt.xlabel('Pr[Correct | Bad]')
plt.ylabel('Pr[Correct | Good]')
plt.savefig('figures/logOddsContour.pdf')

# Overlay questions to be removed, as determined by ECM algorithm
arrRm = np.loadtxt('results/emMartianRemove2013_03_18_15_06_32.txt', dtype = np.int )

oddsDict = {}
toDict = lambda x: ( x[2],( x[0], x[1] ) )
oddsDict = dict( map( toDict, oddsList ) )
rmData = np.asarray(list( oddsDict[str(i)] for i in arrRm ) )

plt.plot( rmData[:,0],rmData[:,1],'y.', alpha = 0.9 )
plt.savefig('figures/logOddsContour_post.pdf')

# Plot rank-ordered log odds
plt.figure(figsize = (5,4))

#plt.errorbar(np.arange(0,logOdds.shape[0]),np.sort(logOdds,order='lO')['lO'],yerr=np.sort(logOdds,order='lO')['se'],fmt='g.',alpha=0.9)
plt.plot(np.sort(logOdds['lO']),'r.-',alpha=0.4)
plt.ylabel('log odds ratio')
plt.xlabel('rank order')
plt.savefig('figures/logOdds.pdf')


# Load evolution of delta estimates
deltas = np.loadtxt( 'results/emMartianDelta2013_03_18_15_06_32.txt', delimiter = ',', dtype = np.float )

# Plot rank-ordered estimated deltas
plt.figure(figsize = (5,4))

D = deltas.shape[1]

( plt.plot( np.arange(0,D), np.sort( deltas[-1,:] ), 'b.', 
            np.arange(0,D), np.ones([D,]), 'r-',alpha=0.4 ) )

plt.ylabel('estimated delta_i''s')
plt.xlabel('rank order')
plt.savefig('figures/deltas_rankOrder.pdf')

# Plot histogram of estimated deltas

plt.figure(figsize = (5,4))

plt.hist(deltas[-1,:],bins = 30, color = 'green')

plt.ylabel('count')
plt.xlabel('delta bins')
plt.savefig('figures/deltas_hist.pdf')

# Plot evolution of a random set of delta estimates to show convergence

plt.figure(figsize = (5,4))

# number of randomly selected deltas to display
dd = 50

# number of iterations to display
iters = 30

# randomly selected deltas to display
ind = np.random.random_integers(0,D-1,dd)

plt.plot( np.vstack( ( np.ones([1,dd]), deltas[0:iters,ind] ) ) )

plt.xlabel('iteration')
plt.ylabel('delta_i evolution')
plt.savefig('figures/deltas_evol.pdf')


