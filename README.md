# Gaussian Quadrature

This repository contains implementations of High-Order Gaussian quadrature methods for numerical approximation of continous integration, including:

$$
\int_{a}^{b} f(x)\,dx \approx \sum_{i=1}^{N} w_i\,f(x_i)
$$

∫ₐᵇ f(x) dx ≈ Σᵢ₌₁ᴺ wᵢ f(xᵢ)

- Gauss–Legendre  
- Gauss–Radau (left and right)  
- Gauss–Lobatto  

Each method computes quadrature nodes and weights on a general interval \([a, b]\) where a is the integration lower limit and a is the Integration upper limit, using a linear mapping from the standard domain \([-1, 1]\).

## Usage

- N: number of quadrature nodes  
- a: lower bound  
- b: upper bound 


```python
import gaussian_quad as gq

nodes, weights = gq.gauss_legendre_nodes_weights(N, a, b)
f = lambda x: x**2 
Integral = (weights * f(nodes)).sum()