## AFFINE.PY

import numpy as np
import matplotlib.pyplot as plt

## Formatting for plots
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'serif','serif':['Computer Modern'],'size':22})

## User-defined settings: change me!

D = 2						## Dimensionality of the parameter space
a = 2.						## Step-size parameter (see arxiv:1202.3665)
nwalkers = 50				## Number of walkers
nmoves = 1000				## Number of steps to iterate through
output = 'outputAI.txt'	## Output file name

cov = np.identity(D)			## Define covariance matrix of the likelihood
inputstd = np.sqrt(np.array([cov[i,i] for i in range(D)]))	## Calc. marginalised widths
data = np.array([0. for i in range(D)])		## Define mean of the likelihood
prange = np.array([[-10.,10.] for i in range(D)]).T  ## Define prior range
icov = np.linalg.inv(cov)

print 'INPUT MEANS:'
print data
print 'INPUT STD:'
print inputstd
print 'INPUT COVMAT:'
print cov

print 'Number of parameters =', D
print 'Number of walkers =', nwalkers

# Choose an initial set of positions for the walkers.
# e.g. a small ball around the max likelihood point
# p = np.array([[data+0.01*np.random.randn(D)*(prange[1,:]-prange[0,:]) for i in range(nwalkers)]])
# or e.g. a small ball around (-5,-5) so we can see the migration fo the walkers. 
p = np.array([[[-5,-5]+0.1*np.random.randn(D)*(prange[1,:]-prange[0,:]) for i in range(nwalkers)]])

## ------------------------------------------------------ ##

## Modify this function to call an arbitrary likelihood
def lnprob(x, mu, invc):
    diff = x-mu
    return -np.dot(diff,np.dot(invc,diff))/2.0

## Generate a random variable z to determine step-size of the walker
def density():
	u = np.random.random()
	return ((u*(a-1.)+1.)**2.)/a

## A function to determine whether or not the new 
## sample point is withinthe prior range.
def outrange(test):
	outrange=False
	for j in range(D):
			if test[j]>prange[1][j] or test[j]<prange[0][j]: outrange=True
	return outrange

## ------------------------------------------------------ ##

## Now let's move the walkers!
## Do nmoves for each walker, using the other walkers' positions to update.
move_num = 0
for move_num in range(nmoves-1):
	newpos = []
	#if move_num % 1000 == 0: print move_num
	for walker in range(nwalkers):
		orange = True
		while(orange == True):
			cur_walker = p[move_num][walker]			
			ran = np.random.random_integers(nwalkers)-1
			while walker == ran: ran = np.random.random_integers(nwalkers)-1
			rand_walker = p[move_num][ran]
			z = density()
			y = rand_walker + z*(cur_walker - rand_walker)
			orange = outrange(y)
			#orange = False
		q = (D - 1)*np.log(z) + lnprob(y, data, icov) - lnprob(cur_walker, data, icov)
		r = np.log(np.random.random())
		if r <= q: newpos.append(y)
		else: newpos.append(cur_walker)
	newpos = np.array(newpos)
	
	## p has the shape (nmoves, nwalkers, D) 
	p = np.append(p, [newpos], axis = 0)
	
	## y has the shape (nmoves * nwalkers, D)
	y = np.reshape(p, (np.shape(p)[0]*np.shape(p)[1], -1))
	
	move_num += 1

## Save the positions of all walkers at each iteration
## into an output text file as specified above
np.savetxt(output, y, delimiter = '\t', fmt = '% .5f')

## Calculate mean position of walkers at each iteration.
mean = np.mean(p, axis = 1)

## Plot the mean walker position at each iteration.
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.scatter(mean.T[0], mean.T[1])
plt.show()

## Plot the position of all the walkers at final iteration
#plt.xlim(-10,10)
#plt.ylim(-10,10)
#plt.scatter(p[-1].T[0], p[-1].T[1])
#plt.show()



