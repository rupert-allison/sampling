## NESTSAMP.PY

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

## Format labels to Latex font
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'serif','serif':['Computer Modern'],'size':22})

N=300 			## Size of active set
D=2			## Number of parameters
f=1.06 			## Expansion factor for the ellipsoid

c = np.identity(D) 		## Target distribution covariance (toy-model Gaussian example)
data=[0 for i in range(D)] 	## Target mean
invc=np.linalg.inv(c)
detc=np.linalg.det(c)

prange=[[-10,10] for i in range(D)] ## Set the prior range

## Model functional form
def model(samplpnts):
	func=samplpnts 
	return func

## Likelihood (toy-model Gaussian)
def LL(samplpnts):
	const=np.log(((2*np.pi)**D)*detc)
	dev=model(samplpnts)-data[:D]
	L=np.array([np.dot(dev[j],np.dot(invc,dev[j])) for j in \
	range(len(samplpnts))])
	return -0.5*L
	
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

x_mem=np.array([[] for i in range(D)]).T ## Define the memory arrays
L_mem=np.array([])

x=[] ## Define the first initial active set of N points
for i in range(D):
	arg=(prange[i][1]-prange[i][0])*np.random.uniform(size=N)+prange[i][0]
	x.append(arg)	

x=np.array(x).T

Levals=0 		## Set counter to zero. 
L=LL(x)			## L now holds the log-likelihood values of the active set
Levals += N

deltalogZ=1. 		# Test statistic determines when to stop the algorithm
evidence=0.		# Evidence accumulated is currently zero
i=0			# Set iteration counter to zero
testfail=0
count=1
Levals=0
oprior=0

s = np.exp(0.1) - 1
while (deltalogZ>s):
	T,cent,el=covmat(x)  	## Gives the covariance matrix and centroid and ellipse
	index_min=np.argmin(L)
	x_mem=np.append(x_mem,[x[index_min]],axis=0) 	## Add sample point to x_mem
	L_mem=np.append(L_mem,L[index_min]) 		## Add sample point to L_mem
	
	x=np.delete(x,index_min,axis=0) 	## Remove the element from the active set x.
	L=np.delete(L,index_min) 		## Remove the element from the active set L

	ytrail=drawsamp(T,cent)
	while outrange(ytrail)==True:
		oprior+=1
		ytrail=drawsamp(T,cent)
	Ltrail=LL(np.array([ytrail]))
	Levals+=1
	while (Ltrail<=L_mem[i]):
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
	evidence+=np.exp(L_mem[i])*w(i+1)
	if evidence>0.: deltalogZ=(priorvol(i)*np.exp(Lmax))/evidence
	
	M=i ## define M as the number of iterations until convergence
	i+=1


print 'Number of iterations =',M
print 'Number of likelihood evaluations =', Levals
print 'Number which failed test =', testfail
print 'Fraction of failures = %.3f' % float(testfail*1./(testfail+M))
print 'Number outside prior range =',oprior

## Append the active sets to the memory sets

x_mem=np.append(x_mem,x,axis=0)
L_mem=np.append(L_mem,L)

pi=np.zeros(M+1+N) ## Initialise an array of weights 
for j in range(M+1):
	pi[j]=np.exp(L_mem[j])*w(j+1)
for j in range(M+1,M+1+N):
	pi[j]=np.exp(L_mem[j])*priorvol(j)/N

mean=np.average(x_mem,weights=pi,axis=0)
meascov=np.zeros((D,D))
for i in range(D):
	for j in range(D):
		meascov[i][j]=np.average((x_mem.T[i]-mean[i])*\
			(x_mem.T[j]-mean[j]),weights=pi)

print '------------------'
print 'Mean of distribution ='
print mean
print '------------------'
print 'Covariance of distribution ='
print meascov
print '------------------'
