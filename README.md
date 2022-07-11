# Exploration into Reducing Uncertainty in Inverse Problems Using Markov Chain Monte Carlo Methods
## Undergraduate Honors Thesis, Department of Mathematics at Dartmouth College
### Jiahui (Jack) Zhang
### Supervised by Dr. Anne Gelb
### June 3, 2020

## Summary

This thesis developed a method to ascertain the unknown recovery and uncertainty quantification
of a one-dimensional inverse problem through the Bayesian framework, specifically by using the
Metropolis-Hastings Algorithm. However instead of using the standard (unweighted) sparsity prior
to create the posterior probability density, a new weighted sparsity prior was generated using the
variance based joint sparsity (VBJS) method. The proposed weights are intended to variably penalize
distinct regions of the domain depending on the variance of the unknown samples in the edge
domain. By imposing these weights on the penalty in the edge domain,
this thesis demonstrated that marked improvements can be made to the convergence properties of
MCMC for a one-dimensional inverse problem. In turn, this lead to improved computational efficiency,
as fewer iterations are needed. The new method also yielded tighter confidence intervals
constructed from the Markov chains constructed by the Metropolis-Hastings algorithm.

## code Included

Code for solving inverse problem with additive noise using Markov chain Monte Carlo methods. In the code, a 1D function is used as the unknown for a Y=AX+E forward model. The code approximates X using MCMC (Metropolis Hastings) with the Weighted (VBJS) ell_1 prior.
