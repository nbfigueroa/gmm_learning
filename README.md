# gmm_learning

My little toolbox for learning Gaussian Mixture Models with different inference methods in MATLAB. The derivations of each inference method are thorougly described in:

[1] Figueroa, N. and Billard, A. "Introduction to Bayesian Non-Parametrics for Gaussian Mixture Models and Hidden Markov Models" EPFL TECH-REPORT #XYZ.

#### Finite Gaussian Mixture Model Learning with EM
The implementation of the standard EM (Expectation-Maximization) method  + Model Selection (BIC/AIC) method for finite GMM parameter estimation is based on the implementation in [ML_toolbox](https://github.com/epfl-lasa/ML_toolbox). 

You can test it with the following script:  ```demo_gmm_em.m```.

#### Bayesian Gaussian Mixture Model Learning with Collapsed Gibbs Sampling
The implementation of the Bayesian GMM collapsed gibbs sampler was losely based on the following [tutorial](http://www.robots.ox.ac.uk/~fwood/teaching/AIMS_CDT_ML/homework/HW_3_sampling/). For implementation details, refer to [1].

You can test it with the following script: ```demo_bgmm_gibbs.m```

#### Bayesian Non-Parametric (BNP) Gaussian Mixture Model Learning with Collapsed Gibbs Sampling
Download the following toolboxes and make sure it's in the MATLAB path: [some other link](). These toolboxes provide code for inference of the **DP-GMM** (Dirichlet Process), a realization of the Infinite Gaussian Mixture Model, which enable one to discover the number of Gaussian functions from the data, rather than doing model selection, hence the name *non-parametric inference*:
- MCMC Inference for DP-GMM [5]:     bli

You can test them with the following script: ```demo_dpgmm_gibbs.m```

#### Dynamical System Learning Estimation via GMM
Here, you can compare a GMM learned through either of the previous inference methods (EM, Sampler for Bayesian GMM with fixed K, Sampler for BNP GMM with unknown K) vs. SEDS [2] for a set of 3D trajectories, and simulate the learned dynamics.


-
#### Implementation and toolbox References:
- [1] Figueroa, N. and Billard, A. "Introduction to Bayesian Non-Parametrics for Gaussian Mixture Models and Hidden Markov Models" EPFL TECH-REPORT #XYZ.
- [2] [ML_toolbox](https://github.com/epfl-lasa/ML_toolbox) Matlab toolbox used for teaching machine learning techniques at EPFL by N. Figueroa among others. 
- [3] [SEDS](https://bitbucket.org/khansari/seds) Stable Estimator of Dynamical Systems with GMM by M. Khansari

