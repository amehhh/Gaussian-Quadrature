import numpy as np 
from numpy.polynomial.legendre import Legendre
from numpy.polynomial.legendre import leggauss

# The quadrature implementations for continous Integrals using linear mappings are based on the paper:
#
# Garg, D., Patterson, M. A., Francolin, C., Darby, C. L.,
# Huntington, G. T., Hager, W. W., & Rao, A. V.
# "Direct trajectory optimization and costate estimation of
# finite-horizon and infinite-horizon optimal control problems
# using a Radau pseudospectral method."
#
# https://link.springer.com/article/10.1007/s10589-009-9291-0
# https://arc.aiaa.org/doi/epdf/10.2514/1.33117
# https://personal.ntu.edu.sg/lilian/CiCp09Review.pdf

tol = 1e-12

def gauss_legendre_nodes_weights(N, a=0.0, b=1.0):

    """
    N-point Gauss–Legendre quadrature on the interval [a,b] (excluding endpoints).
    Returns nodes t and weights w via a linear maping.
    """
    x, w = leggauss(N)                      # returns nodes/weights on the interval [-1,1]
    t_nodes = 0.5*(b-a)*(x+1) + a           # map nodes to the interval of integration [a,b]
    weights = 0.5*(b-a) * w                 # weights for scaling
    return t_nodes, weights

def gauss_radau_left(N, a=-1.0, b=1.0):
        
    """
    Left Gauss–Radau quadrature on the interval [a,b] including the end-point x = -1.
    Left Gauss_Radau quadrature is Based on:
        roots of (P_{N-1}(x) + P_N(x)) = 0
        w_i = (1 - x_i) / [N * P_{N-1}(x_i)]^2 - Interior weights for other nodes
        w_0 = 2 / N^2 - end point weight at the fixed node x=-1
    """
    
    # Legendre polynomials
    Pn   = Legendre.basis(N)
    Pn_1 = Legendre.basis(N-1)

    # Internal nodes: roots of the sum of legendre polynomials P_{N-1}(x) + P_N(x)
    roots = (Pn_1 + Pn).roots()

    # Remove any root numerically equal to -1
    roots = roots[roots > -1 + tol]

    # Combine with exact endpoint -1
    x_nodes = np.concatenate(([-1.0], roots))

    # Compute weights
    w = np.zeros_like(x_nodes)
    w[0]  = 2.0 / (N**2)
    w[1:] = (1 - x_nodes[1:]) / ((N * Pn_1(x_nodes[1:]))**2)

    # Map from nodes [-1,1] → [a,b] the integration interval
    t_nodes = 0.5 * (b - a) * (x_nodes + 1) + a
    weights = 0.5 * (b - a) * w
    return t_nodes, weights

def gauss_radau_right(N, a=-1.0, b=1.0):

    """
    Right Gauss–Radau quadrature on the interval [a,b] including at the end-point x = +1.
    Right Gauss-Radau is based on:
        roots of (P_N(x) - P_{N-1}(x)) = 0
        w_i = (1 + x_i) / [N * P_{N-1}(x_i)]^2 - Interior weights 
        w_N = 2 / N^2 - Endpoint weights for other node at the fixed node x = +1
    """

    # Legendre polynomials
    Pn   = Legendre.basis(N)
    Pn_1 = Legendre.basis(N-1)

    # Internal nodes: roots of P_N(x) - P_{N-1}(x)
    poly = Pn - Pn_1
    roots = np.sort(poly.roots())

    #  Remove any root numerically equal to +1
    roots = roots[roots < 1 - tol]

    #Combine with exact endpoint +1
    x_nodes = np.concatenate((roots, [1.0]))

    # Compute weights
    w = np.zeros_like(x_nodes)
    w[-1] = 2.0 / (N**2)
    w[:-1] = (1 + x_nodes[:-1]) / ((N * Pn_1(x_nodes[:-1]))**2)

    # Map from [-1,1] → [a,b] the integral interval
    t_nodes = 0.5 * (b - a) * (x_nodes + 1) + a
    weights = 0.5 * (b - a) * w
    return t_nodes, weights

def gauss_lobatto_nodes_weights(N, a=-1.0, b=1.0):

    """
    Gauss-Lobatto quadrature inside the interval [a,b], with the both endpoints x=-1 and x=+1.
    Based on the Roots of the derivative of the Legendre polynomial P_{N-1}'(x) with endpoints x=-1 and x=1 concatenated before mapping 
    w_0 = w_N = 2.0 / (N * (N -1 )) - Endpoint Weights 
    w_i = 2 / [N (N - 1) (P_{N-1}(x_i))^2] - Interior weights
    """

    if N < 2:
        raise ValueError("Need at least 2 Lobatto nodes")

    # Interval nodes: Roots of the derivate of P(N-1) or P'(N-1) 
    Pn_1 = Legendre.basis(N-1)
    x_int = Pn_1.deriv().roots()  
    x_nodes = np.concatenate(([-1.0], x_int, [1.0])) # Added the endpoint nodes

    w = np.zeros(N)
    w[0] = w[-1] = 2.0 / (N * (N - 1))

    # Evaluate the polynomial at all points once for interior weights
    Pn_1_vals = Pn_1(x_int) 
    w[1:-1] = 2.0 / (N*(N-1) * (Pn_1_vals ** 2))

    # Map to [a, b], the integral domain
    t_nodes = 0.5 * (b - a) * (x_nodes + 1) + a
    weights = 0.5 * (b - a) * w
    return t_nodes, weights


# The following functions are for infite horizon mapping on the domain [a,∞] and different types of mappings

def gauss_radau_left_infinite(N):


    """
    Left Gauss–Radau quadrature on the interval [a,∞] 
    Left Gauss_Radau quadrature is Based on:
        roots of (P_{N-1}(x) + P_N(x)) = 0
        w_i = (1 - x_i) / [N * P_{N-1}(x_i)]^2 - Interior weights for other nodes
        w_0 = 2 / N^2 - end point weight at the fixed node x=-1
        Infinite mapping to the domain 0 to ∞
    """
    
    # Legendre polynomials
    Pn   = Legendre.basis(N)
    Pn_1 = Legendre.basis(N-1)

    # Internal nodes: roots of the sum of legendre polynomials P_{N-1}(x) + P_N(x)
    roots = (Pn_1 + Pn).roots()

    # Remove any root numerically equal to -1
    roots = roots[roots > -1 + tol]

    # Combine with exact endpoint -1
    x_nodes = np.concatenate(([-1.0], roots)) 

    # Compute weights
    w = np.zeros_like(x_nodes)
    w[0]  = 2.0 / (N**2)
    w[1:] = (1 - x_nodes[1:]) / ((N * Pn_1(x_nodes[1:]))**2)

    # Infinite-horizon mapping τ → t
    t_nodes = (1 + x_nodes) / (1 - x_nodes)

    # Transformed weights: multiply by dt/dτ
    dt_dtau = 2.0 / (1 - x_nodes)**2
    weights = w * dt_dtau

    return t_nodes, weights

def gauss_radau_left_infinite_logmapping(N):

    """
    Left Gauss–Radau quadrature on [-1,1] → [0,∞) using the nodes
    from roots of the legendre polynomial P_N + P_{N-1}, mapped using t = log(4 / (1 - tau)^2) and dt/dtau = 2 /(1-tau)
    """

    # Legendre polynomials
    Pn   = Legendre.basis(N)
    Pn_1 = Legendre.basis(N-1)

    # Internal nodes: roots of the sum of legendre polynomials P_{N-1}(x) + P_N(x)
    roots = (Pn_1 + Pn).roots()

    # Remove any root numerically equal to -1
    roots = roots[roots > -1 + tol]

    # Combine with exact endpoint -1
    x_nodes = np.concatenate(([-1.0], roots))

    # Compute weights
    w = np.zeros_like(x_nodes)
    w[0]  = 2.0 / (N**2)
    w[1:] = (1 - x_nodes[1:]) / ((N * Pn_1(x_nodes[1:]))**2)

    # Infinite-horizon log mapping τ → t with the mapping t = log(4 / (1 - tau)^2)
    t_nodes = np.log(4.0 / (1 - x_nodes)**2)

    dt_dtau = 2.0 / (1 - x_nodes)
    weights = w * dt_dtau

    return t_nodes,weights



    

