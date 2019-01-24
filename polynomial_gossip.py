import numpy as np
import scipy.spatial
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs
from scipy.sparse.csgraph import shortest_path, connected_components
import matplotlib.pyplot as plt

# BUILDING OF GRAPHS AND GOSSIP MATRICES

def grid_gossip(values):
    # applies the laplacian in a d-dimensional grid
        d = len(values.shape)
        gossiped = np.zeros(values.shape)
        for dim in range(d):
            gossiped += np.roll(values,1,axis=dim) \
            + np.roll(values,-1,axis=dim)
        gossiped /= 2*d
        return gossiped
    
def gossip_matrix_from_adjacency_matrix(A) :
    sums = sum(A.T)
    np.place(sums,sums == 0, 1)
    B = np.diag(1/sums).dot(A)
    C = np.minimum(B,B.T)
    return C + np.diag(1-sum(C.T))

def build_3D_graph(n):
    l = int(np.round(n**(1/3)))
    A = np.zeros((n,n))
    for i in range(l) :
        for j in range(l) :
            for k in range(l) : 
                if j != l-1 :
                    A[i*l**2+j*l+k,i*l**2+(j+1)*l+k] = 1
                if j != 0 :
                    A[i*l**2+j*l+k,i*l**2+(j-1)*l+k] = 1
                if i != l-1 :
                    A[i*l**2+j*l+k,(i+1)*l**2+j*l+k] = 1
                if i != 0 :
                    A[i*l**2+j*l+k,(i-1)*l**2+j*l+k] = 1
                if k != 0 : 
                    A[i*l**2+j*l+k,i*l**2+j*l+(k-1)] = 1
                if k != l-1 :
                    A[i*l**2+j*l+k,i*l**2+j*l+(k+1)] = 1
    W = gossip_matrix_from_adjacency_matrix(A)
    return csr_matrix(W)

def build_2D_percolation_graph(n,p):
    l = int(np.round(np.sqrt(n)))
    A = np.zeros((n,n))
    for i in range(l) :
        for j in range(l) :
            if j != l-1 and np.random.uniform() < p :
                A[i*l+j,i*l+(j+1)] = 1
                A[i*l+(j+1),i*l+j] = 1
            if i != l-1 and np.random.uniform() < p :
                A[i*l+j,(i+1)*l+j] = 1
                A[(i+1)*l+j,i*l+j] = 1
    W = csr_matrix(np.identity(n)+A/4-np.diag(A@np.ones(n))/4)
    return W

def build_3D_percolation_graph(n,p):
    l = int(np.round(n**(1/3)))
    A = np.zeros((n,n))
    for i in range(l) :
        for j in range(l) :
            for k in range(l) :
                if j != l-1 and np.random.uniform() < p :
                    A[i*l**2+j*l+k,i*l**2+(j+1)*l+k] = 1
                    A[i*l**2+(j+1)*l+k,i*l**2+j*l+k] = 1
                if i != l-1 and np.random.uniform() < p :
                    A[i*l**2+j*l+k,(i+1)*l**2+j*l+k] = 1
                    A[(i+1)*l**2+j*l+k,i*l**2+j*l+k] = 1
                if k != l-1 and np.random.uniform() < p :
                    A[i*l**2+j*l+k,i*l**2+j*l+(k+1)] = 1
                    A[i*l**2+j*l+(k+1),i*l**2+j*l+k] = 1
    W = csr_matrix(np.identity(n)+A/6-np.diag(A@np.ones(n))/6)
    return W

def build_2D_graph(n):
    return build_2D_percolation_graph(n,1)

def build_3D_graph(n):
    return build_3D_percolation_graph(n,1)
    
def adjacency_and_positions_rdm_geom_graph(n,d,r,cyclic=False) :
    positions = np.random.uniform(size=(n,d))
    if cyclic :
        distances1 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(positions))
        positions[:,0] += 1/2
        positions,_ = np.modf(positions)
        distances2 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(positions))
        positions[:,1] += 1/2
        positions,_ = np.modf(positions)
        distances3 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(positions))
        positions[:,0] += 1/2
        positions,_ = np.modf(positions)
        distances4 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(positions))
        distances = np.minimum(distances1,np.minimum(distances2,np.minimum(distances3,distances4)))
    else :
        distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(positions))
    A = (distances <= r)
    np.fill_diagonal(A,0)
    return A,positions

def adjacency_random_regular_graph(n,d) :
    # n or d must be even !
    # note that there can be some multiple edges or loops in the output graph
    perm = np.random.permutation(n*d)
        
    A = lil_matrix((n,n))
    for i in range(n*d//2) :
        A[perm[2*i]//d,perm[2*i+1]//d] += 1
        A[perm[2*i+1]//d,perm[2*i]//d] += 1
    return A

def largest_component(W, return_indices=False):
    _, components = connected_components(W, directed = False)
    counts = np.bincount(components)
    largest = np.argmax(counts)
    sel = (components == largest)
    if return_indices :
        return (W[np.ix_(sel, sel)], sel)
    else :
        return W[np.ix_(sel, sel)]

# FUNCTIONS FOR THE COMPARISON OF GOSSIP METHODS - PLOTS OF THE PAPER

def do_the_gossip(methods, T, W, initial_values, goal="average"):
    """
    methods: list composed of some of the following elements: 
    "simple", "shift-register", "splitting", "local averaging", 
    ("jacobi",d) where d is the parameter of the Jacobi iteration.
    goal: value that the estimators are compared to, can be set
    for instance to the mean of the distribution of the initial 
    values. Default is the average of the initial values. 
    """
    
    if goal == "average":
        average = np.sum(initial_values)/len(initial_values)
    else:
        average = goal
        
    # useful for the tuning of some methods
    def _need_spectral_gap(method):
        if method == "shift-register" or method == "splitting":
            return True
        elif type(method) == tuple and method[0] == "jacobi-gap":
            return True
        else: 
            return False
    if True in map(_need_spectral_gap,methods): 
        eigenvalues = eigs(W,k=2,return_eigenvectors = False, 
                           which='LR')
        radius = eigenvalues[0]
        
    results = {}
        
    for method in methods:
        variances = np.zeros(T)
        if method == "local averaging":
            n = len(initial_values)
            distances = shortest_path(W != 0, directed = False)
            for t in range(T):
                variances[t] = sum((((distances <=t)@initial_values)
                                    /((distances <= t)@np.ones(n))
                          -sum(initial_values)/n)**2)/n
        else:
            if method == "simple":
                g = SimpleGossip(initial_values, lambda x: W.dot(x))
            elif method == "shift-register":
                g = ShiftRegisterGossip(initial_values,
                    (8-4*np.sqrt(4-(1+radius)**2))/(1+radius)**2, 
                                        lambda x: W.dot(x))
            elif method == "splitting":
                g = SplittingGossip(initial_values, 
                                    2/(1+np.sqrt(1-radius**2)), 
                                    lambda x: W.dot(x))
            elif type(method) == tuple and method[0] == "general-jacobi":
                alpha = method[1]
                beta = method[2]
                coeffs = lambda t: coeffs_Jacobi(alpha,beta,t)
                g = PolynomialGossip(initial_values, coeffs, 
                                     lambda x: W.dot(x))
            elif type(method) == tuple and method[0] == "jacobi":
                d = method[1]
                coeffs = lambda t: coeffs_Jacobi(d/2,0,t)
                g = PolynomialGossip(initial_values, coeffs, 
                                     lambda x: W.dot(x))
            elif type(method) == tuple and method[0] == "jacobi-gap":
                d = method[1]
                coeffs = lambda t: coeffs_Jacobi_gap(d/2,0,1-radius,t)
                g = NormalizePolynomialGossip(initial_values, coeffs, 
                                     lambda x: W.dot(x))
            elif method == "best polynomial gossip":
                g = ParameterFreePolynomialGossip(initial_values,
                                                 lambda x:W.dot(x))
            elif type(method) == tuple and method[0] == "message-passing":
                d = method[1]
                g = PolynomialGossip(initial_values, 
                                     lambda t: coeffs_Kesten_McKay(d, t), 
                                     lambda x: W.dot(x))
            else:
                raise NameError("The method "+str(method)+" doesn't exist.")
            for t in range(T) : 
                variances[t] = np.linalg.norm(g.mu-average)**2\
                                    /g.mu.size
                g.gossip()
        results[method] = variances
    
    return results


def method_to_label_color_linestyle(method):
    if method == 'simple':
        return ("Simple Gossip","red","--")
    elif method == "shift-register":
        return ("Shift-Register Gossip","green",(0, (3, 1, 1, 1, 1, 1)))
    elif method == "local averaging":
        return ("Local Averaging","grey",":")
    elif method == "splitting":
        return ("Min-Sum Splitting","blue",(0, (3, 1, 1, 1, 1, 1)))
    elif type(method) == tuple and method[0] == "general-jacobi":
        n_color += 1
        return (r"$\alpha =$"+str(method[1])+', '+r"$\beta =$"+str(method[2]),
                "C{}".format(n_color),"-")
    elif type(method) == tuple and method[0] == "jacobi":
        return ("Jacobi Polynomial Iteration",
                "orange","-")
    elif type(method) == tuple and method[0] == "jacobi-gap":
        return ("Jacobi Pol. It. with Spectral Gap", 
                "magenta", "-.")
    elif method == "best polynomial gossip":
        return ("Parameter Free Pol. It.",
               "blue", (0, (1, 1)))
    elif type(method) == tuple and method[0] == "message-passing":
        return ("Message Passing Gossip", 
                "cyan", "solid")
    else:
        return ("","black","-")
    
def plot_curves(results, T, legend=True, logscale=False, xmin=0, xmax=None, ymin=0, ymax=1.05, figsize=(6,5), ylabel=r"$\Vert x^t-\bar{\xi}\mathbb{1}\Vert_2/\sqrt{n}$", methods = None):
    ts = [t for t in range(T)]
    fig, ax = plt.subplots(figsize=figsize)
    linewidth = 3
    if xmax == None:
        xmax = T
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    if methods == None:
        methods = results
    for method in methods:
        (label,color,linestyle) = \
        method_to_label_color_linestyle(method)
        if logscale:
            ax.semilogy(ts, np.sqrt(results[method]), 
                    color, label=label,linestyle=linestyle,
                    linewidth=linewidth)
        else:
            ax.plot(ts, np.sqrt(results[method]), 
                    color, label=label,linestyle=linestyle,
                    linewidth=linewidth)
    if legend:
        ax.legend(loc='upper right',handlelength=5, fontsize=13)
    ax.yaxis.set_tick_params(direction = 'out', length = 10, 
                             width = 2,labelsize = 15)
    ax.xaxis.set_tick_params(direction = 'out', length = 10, 
                             width = 2, labelsize = 15)
    plt.xlabel(r"$t$",fontsize=16)
    plt.ylabel(ylabel,fontsize=16)
    return fig

# GENERIC POLYNOMIAL GOSSIP METHOD 
        
class PolynomialGossip() :
    # polynomial gossip method with given recursion parameters
    def __init__(self,initial_values,coeffs,gossip_operation=grid_gossip):
        self.name = "Polynomial Gossip"
        self.gossip_operation = gossip_operation
        self.coeffs = coeffs
        self.mu = np.copy(initial_values)
        self.t = 0
    def gossip(self):
        (a,b,c) = self.coeffs(self.t)
        if self.t == 0 :
            mu_new = a*self.gossip_operation(self.mu) + b*self.mu
        else :
            mu_new = (a*self.gossip_operation(self.mu) + b*self.mu 
                    -c*self.mu_prec)
        self.mu_prec = self.mu
        self.mu = mu_new
        self.t += 1
        
class NormalizePolynomialGossip():
    # polynomial gossip method with given recursion parameters, 
    # but we do not assume that the recursion gives orthogonal 
    # polynomials normalized such that P_t(1) = 1. 
    # Thus the algorithm does the normalization itself. 
    def __init__(self,initial_values,coeffs,gossip_operation=grid_gossip):
        self.name = "Polynomial Gossip"
        self.gossip_operation = gossip_operation
        self.coeffs = coeffs
        self.mu = np.copy(initial_values)
        self.x = np.copy(initial_values)
        self.y = 1
        self.t = 0
    def gossip(self):
        (a,b,c) = self.coeffs(self.t)
        if self.t == 0 :
            x_new = a*self.gossip_operation(self.x) + b*self.x
            y_new = a*self.y + b*self.y
        else :
            x_new = (a*self.gossip_operation(self.x) + b*self.x
                    -c*self.x_prec)
            y_new = (a*self.y + b*self.y 
                    -c*self.y_prec)
        self.x_prec = self.x
        self.x = x_new
        self.y_prec = self.y
        self.y = y_new
        self.mu = self.x / self.y
        self.t += 1
        
class ParameterFreePolynomialGossip():
    def __init__(self,initial_values,gossip_operation=grid_gossip):
        self.gossip_operation = gossip_operation
        self.mu = np.copy(initial_values)
        self.t = 0
    def gossip(self):
        if self.t == 0:
            Wmu = self.gossip_operation(self.mu)
            b = -(self.mu@Wmu - Wmu@Wmu) / (self.mu@self.mu-self.mu@Wmu) 
            mu_new = (Wmu + b*self.mu) / (1+b)
        else:
            Wmu = self.gossip_operation(self.mu)
            Wmu_prec = self.gossip_operation(self.mu_prec)
            b = -(self.mu@Wmu - Wmu@Wmu) / (self.mu@self.mu - self.mu@Wmu)
            c = ((Wmu@self.mu_prec - Wmu@Wmu_prec) 
                 / (self.mu_prec@self.mu_prec - self.mu_prec@Wmu_prec))
            mu_new = (Wmu + b*self.mu - c*self.mu_prec) / (1+b-c)
        self.mu_prec = self.mu
        self.mu = mu_new
        self.t += 1
        
# OTHER GOSSIP METHODS

class SimpleGossip() :
    def __init__(self,initial_values,gossip_operation=grid_gossip):
        self.name = "Simple Gossip"
        self.gossip_operation = gossip_operation
        self.mu = np.copy(initial_values)
    def gossip(self):
        self.mu = self.gossip_operation(self.mu)
        
class ShiftRegisterGossip() :
    def __init__(self,initial_values,omega,gossip_operation=grid_gossip):
        self.name = "Shift-Register Gossip"
        self.gossip_operation = gossip_operation
        self.mu = np.copy(initial_values)
        self.omega = omega
        self.t = 0
    def gossip(self):
        self.t += 1
        if self.t == 1:
            mu_new = self.gossip_operation(self.mu)
        else:
            mu_new = (self.omega * self.gossip_operation(self.mu) 
                  + (1 - self.omega) * self.mu_prec)
        self.mu_prec = self.mu
        self.mu = mu_new

# COEFFICIENTS FOR THE ORTHOGONAL POLYNOMIALS
        
def coeffs_Jacobi(alpha,beta,t):
    if t == 0 : 
        return ((alpha + beta + 2)/(2*(1+alpha)),
                (alpha-beta)/(2*(1+alpha)),
                0) # the c value here isn't used so we do not care
    else :
        return ((2*t+alpha+beta+1)*(2*t+alpha+beta+2)/(2*(t+1+alpha+beta)*(t+1+alpha)),
                (2*t+alpha+beta+1)*(alpha+beta)*(alpha-beta)/(2*(t+1+alpha+beta)*(2*t+alpha+beta)*(t+1+alpha)),
                t*(t+beta)*(2*t+alpha+beta+2)/((t+1+alpha+beta)*(2*t+alpha+beta)*(t+1+alpha)))
    
def coeffs_Kesten_McKay(d,t):
    if t == 0 :
        return (d/(d+1), 1/(d+1), 0)
    else :
        return ((d/(d-1)-2*(d-1)**(-(t+1)))/(1-2/d*(d-1)**(-(t+1))), 
                0, 
                (1/(d-1)-2/d*(d-1)**(-t))/(1-2/d*(d-1)**(-(t+1))))
    
def coeffs_Jacobi_gap(alpha,beta,gap,t):
    # be careful as these coefficients do not produce 
    # polynomials normalized such that P_t(1) = 1
    (a,b,c) = coeffs_Jacobi(alpha,beta,t)
    return (a/(1-gap/2), a/(1-gap/2)*gap/2+b, c)
