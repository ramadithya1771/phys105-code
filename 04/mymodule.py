import math
from matplotlib import pyplot as plt

# let's define the Guassian function with default parameter values:
# Gaussian function https://en.wikipedia.org/wiki/Gaussian_function

def gaus(x,mu=0.,sigma=1.):
    return 1./(sigma*math.sqrt(2.*math.pi)) * math.exp(-(x-mu)**2/(2.*sigma**2))

# let's define a multi-Gaussian function

def multigaus(x,d={'1':{'mu':0.,'sigma':1.}}):
    out = 0.
    for k in d.keys():
        mu = d[k]['mu']
        sigma = d[k]['sigma']
        out += gaus(x,mu,sigma)
    return out

# let's define a derivative function

def df(x,y,x0):
    if x0 < min(x) or x0 > max(x):
        raise Exception('x0 is not within the function x range')
    ind = 0
    while x[ind] < x0: # what's the assumption here?
        ind += 1
    # indices around x0
    ind2 = ind
    ind1 = ind - 1
    return (y[ind2]-y[ind1])/(x[ind2]-x[ind1])

if __name__ == "__main__":
    
    mu = 0.
    sigma = 1.
    x = [i/10*sigma for i in range(-100,100)]
    y = [gaus(X,mu=mu,sigma=sigma) for X in x]
    plt.plot(x,y)
    plt.show()
    
    # let's see what it looks like to have three Gaussians with different standard deviation
    
    gausd = {
        '1':{'mu':-5.,'sigma':1.},
        '2':{'mu':0.,'sigma':0.5},
        '3':{'mu':5.,'sigma':2.}
            }
    yy = [multigaus(X,gausd) for X in x]
    plt.plot(x,yy)
    plt.show()
    
    # let's compute derivatives at x0=-6,5
    x1 = -6
    y1 = multigaus(x1,gausd)
    s1 = df(x,yy,x1)
    x2 = 5
    y2 = multigaus(x2,gausd)
    s2 = df(x,yy,x2)
    
    # let's plot the lines
    plt.plot(x,yy)
    yy1 = [s1*(X-x1)+y1 for X in x] # line crossing (x1,y1)
    yy2 = [s2*(X-x2)+y2 for X in x] # line crossing (x2,y2)
    plt.plot(x,yy1,'k--')
    plt.plot(x,yy2,'k:')
    plt.ylim(0,1)
    plt.scatter([x1,x2],[y1,y2])
    plt.show()

