## NESTSAMP.PY

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

## Format labels to Latex font
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'serif','serif':['Computer Modern'],'size':22})

N=300 					## Size of active set
D=2						## Number of parameters
f=1.06 					## Expansion factor for the ellipsoid
output = 'outputNS.txt' ## Name of the file to contain weights and sample points

c = np.identity(D) ## Target distribution covariance (toy-model Gaussian example)
inputstd = np.sqrt(np.array([c[i,i] for i in range(D)])) ## Calc. marginalised widths
prange=[[-10.,10.] for i in range(D)] ## Set the prior range
data=[0. for i in range(D)] ## Target mean
invc=np.linalg.inv(c)
detc=np.linalg.det(c)

print 'INPUT MEANS:'
print data
print 'INPUT STD:'
print inputstd
print 'INPUT COVMAT:'
print c

## ------------------------------------------------------ ##

## Model functional form
def model(samplpnts):
	func=samplpnts ## Usually the model will have a more complex form
	return func

## Likelihood (toy-model Gaussian)
def LL(samplpnts):
	const=np.log(((2*np.pi)**D)*detc)
	dev=model(samplpnts)-data[:D]
	L=np.array([np.dot(dev[j],np.dot(invc,dev[j])) for j in \
	range(len(samplpnts))])
	return -0.5 * (L + const)
	
def covmat(x):
	
	covar=np.cov(x,bias=1,rowvar=0)	# Covariance of active set
	cent=np.mean(x,axis=0)			# Centriod of active set
	u=x-cent						# Residuals
	covinv=np.linalg.inv(covar)		# Inverse covariance
	eval,R=np.linalg.eigh(covar) 	# Return evals and matrix of evectors
	Cprime=np.diag(eval)			# Diagonal matrix of evals
	Dprime=np.sqrt(Cprime)			# Square root of diagonal matrix 
	g=np.dot(covinv,u.T)
	k=sum([u.T[j]*g[j] for j in range(D)]).max()
	k*=f*f							# Dilate ellisoidal by factor f along each axis
	
	T=np.sqrt(k)*(np.dot(np.dot(R.T,Dprime),R))	## Maps point in the unit 		
												## sphere to ellipsoid

	el=Ellipse(cent,2*np.sqrt(k*eval[0]),2*np.sqrt(k*eval[1]),\
	angle=(180./np.pi)*np.arctan2(R[1,0],R[0,0]),fill=False,color='b', linewidth=4)
	
	return T, cent, el

def drawsamp(T,cent):
	z=np.random.normal(size=D)
	z/=np.sqrt(np.dot(z,z)) ## Rescale z to lie on unit sphere
	unidev=np.random.uniform()  ## Draw a deviate from U[0,1]
	z*=unidev**(1./D) ## rescale z to lie within the unit sphere
	y=np.dot(T,z)+cent ## y is our new sample point	
	return y
		
def priorvol(j):
	return np.exp(-j*1./N)
	
def w(j):
	return 0.5*(priorvol(j-1)-priorvol(j+1))

def outrange(y):
	## A function to determine whether or not the new sample point is within
	## the prior range.
	outrange=False
	for j in range(D):
			if y[j]>prange[j][1] or y[j]<prange[j][0]: outrange=True
	return outrange

## ------------------------------------------------------ ##

## Let's set up active set and the arrays to hold the sample points
## and their likelihood

x_samp=np.array([[] for i in range(D)]).T ## Define the memory arrays
L_samp=np.array([])

x=[] ## Define the first initial active set of N points
for i in range(D):
	arg=(prange[i][1]-prange[i][0])*np.random.uniform(size=N)+prange[i][0]
	x.append(arg)	

x=np.array(x).T

Levals=0 ## Set counter to zero. 
L=LL(x) ## L now holds the log-likelihood values of the active set
Levals += N

deltalogZ=1. 	# Test statistic determines when to stop the algorithm
evidence=0.		# Evidence accumulated is currently zero
i=0				# Set iteration counter to zero

## Define a few counters for evaluating meta-statistics
testfail=0
count=1
Levals=0
oprior=0

## Now let's start the sampling!

s = np.exp(0.1) - 1
while (deltalogZ>s):
	T,cent,el=covmat(x)  ## Gives the covariance matrix and centroid and ellipse
	index_min=np.argmin(L)
	x_samp=np.append(x_samp,[x[index_min]],axis=0) ## Add sample point to x_samp
	L_samp=np.append(L_samp,L[index_min]) ## Add sample point to L_samp
	
	x=np.delete(x,index_min,axis=0) ## Remove the element from the active set x.
	L=np.delete(L,index_min) ## Remove the element from the active set L

	ytrail=drawsamp(T,cent)
	while outrange(ytrail)==True:
		oprior+=1
		ytrail=drawsamp(T,cent)
	Ltrail=LL(np.array([ytrail]))
	Levals+=1
	while (Ltrail<=L_samp[i]):
		testfail+=1
		ytrail=drawsamp(T,cent)
		while outrange(ytrail)==True:
			oprior+=1
			ytrail=drawsamp(T,cent) ## Draw samples until one fits the bill
		Ltrail=LL(np.array([ytrail]))
		Levals+=1

	x=np.append(x,[ytrail],axis=0)
	L=np.append(L,Ltrail,axis=0)
	
	Lmax=L[np.argmax(L)] 
	evidence+=np.exp(L_samp[i])*w(i+1)
	if evidence>0.: deltalogZ=(priorvol(i)*np.exp(Lmax))/evidence
	
	M=i ## define M as the number of iterations until convergence
	i+=1

## Print off some stats
print '------------------'
print 'Number of iterations =', M
print 'Number of likelihood evaluations =', Levals
print 'Number which failed test =', testfail
print 'Fraction of failures = %.3f' % float(testfail*1./(testfail+M))
print 'Number outside prior range =',oprior

## Append the active set to the sample points

x_samp=np.append(x_samp,x,axis=0)
L_samp=np.append(L_samp,L)

## Define the weights for each sample point based on their order
## and likelihood
pi=np.zeros(M+1+N) 
for j in range(M+1):
	pi[j]=np.exp(L_samp[j])*w(j+1)
for j in range(M+1,M+1+N):
	pi[j]=np.exp(L_samp[j])*priorvol(M)/N
	evidence += np.exp(L_samp[j])*priorvol(M)/N

pi /= evidence

## Compute the inferred mean and covariance of the posterior
mean=np.average(x_samp,weights=pi,axis=0)
meascov=np.zeros((D,D))
for i in range(D):
	for j in range(D):
		meascov[i][j]=np.average((x_samp.T[i]-mean[i])*\
			(x_samp.T[j]-mean[j]),weights=pi)

## Print off the lowest-order posterior statistics
print '------------------'
print 'Posterior mean ='
print mean
print '------------------'
print 'Posterior covariance ='
print meascov
print '------------------'

## Save the weights, likelihoods and sample points into output file
master = np.append(np.reshape(pi, (np.shape(pi)[0],1)),np.append(np.reshape(L_samp, (np.shape(L_samp)[0],1)), x_samp, axis = 1) , axis = 1)
np.savetxt(output, master, delimiter = '\t', fmt = '% .5e' )
