## METROP.PY
## A simple Metropolis-Hasting algorithm for a two parameter model
## with uniform priors

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import mlab

## Format labels to Latex font
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'serif','serif':['Computer Modern'],'size': 22})

## Global variables for the initialisation and trial distribution
sigma = np.sqrt(2.4 * 0.04)							## Trial width (isotropic), see arXiv:astro-ph/0405462
x0 = np.array([0.03,-0.03])							## Starting point x0

## Define the log-likelihood
cov = np.array([[0.04,0.01],[0.01,0.01]])				## Covariance of Gaussian likelihood
invcov = np.linalg.inv(cov)								## Inverse covariance, as used in likelihood
def loglike(x): ## Unnormalised!
	chi2 = np.dot(x,np.dot(invcov, x.T))
	return -0.5*chi2

## Sample from trial distribution, given x
def trial(x):
	return x + sigma * np.random.randn(2)

## Define chain and log likelihood lists
chain = []
LL = []

## Append initial position and LL to 
chain.append(x0)
curlike = loglike(x0)
LL.append(curlike)

## Start at x = x0, j is the number of accepted moves
x = x0
j=0
Nsteps = 50000
for i in range(Nsteps):
	y = trial(x)
	triallike = loglike(y)
	r = np.exp(triallike - curlike)
	accprob = min(1, r)
	u = np.random.random()
	if (u < accprob):
		x = y
		curlike = triallike
		LL.append(curlike)
		chain.append(x)
	else:
		LL.append(curlike)
		chain.append(x)
		j+=1

chain = np.array(chain)
LL = np.array(LL)		
print 'Num of steps in chain:', Nsteps
print 'Num of accepted moves:', j

## Plot the autocorrelation function of the chain
## Inspect to determine correlation length (and hence no. of independent samples)
plt.plot(np.correlate(np.array(chain).T[0],np.array(chain).T[0], mode='same'))
plt.show()

## Plot the chain itself
## Check to see if it looks sensible
plt.plot(np.array(chain).T[0])
plt.show()

print '\n---------------------------'
print 'Parameter covariance matrix'
print '---------------------------'
covar = np.cov(chain.T)
print covar

print '\n----------------------------'
print 'Parameter correlation matrix'
print '----------------------------'
rho = np.corrcoef(chain.T)
print rho

## Thin the chain to every M steps for plotting purposes?
M = 100
chain = np.array(chain[0::M])
LL = np.array(LL[0::M])

## Plot the inferred covariance ellipse
cent = np.mean(chain, axis = 0)
u = chain - cent
eval,R=np.linalg.eigh(covar)
covinv=np.linalg.inv(covar)
g=np.dot(covinv,u.T)

k=1
el1=Ellipse(cent,2*np.sqrt(k*eval[0]),2*np.sqrt(k*eval[1]),\
	angle=(180./np.pi)*np.arctan2(R[1,0],R[0,0]),fill=False, color='b', linewidth = 4)

k=2**2
el2=Ellipse(cent,2*np.sqrt(k*eval[0]),2*np.sqrt(k*eval[1]),\
	angle=(180./np.pi)*np.arctan2(R[1,0],R[0,0]),fill=False, color='b', linewidth = 4)

fig=plt.figure(figsize=(10,10)) 
plt.xlabel(r'$\mathbf{\Theta}_1$')
plt.ylabel(r'$\mathbf{\Theta}_2$')
ax=fig.add_subplot(111)
plt.axis([-0.75,0.75,-0.75,0.75])
ax.add_patch(el1)
ax.add_patch(el2)

plt.scatter(chain.T[0],chain.T[1], s=1)
plt.show()

