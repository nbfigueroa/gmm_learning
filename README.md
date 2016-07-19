# gmm_learning

My little toolbox for learning Gaussian Mixture Models with different inference methods in MATLAB.

-
#### Gaussian Mixture Model Learning with EM
If you are only interested in the standard EM (Expectation-Maximization) method [1] + Model Selection (BIC/AIC) you do not need any external toolbox and can go directly to the demo: ```demo_gmm_em.m```.

-
###TODO before the summer!:

### Other Inference Methods
#### Parametric:
Download the following toolbox and make sure it's in the MATLAB path: [MixEst](https://github.com/utvisionlab/mixest). MixEst is a very nice MATLAB toolbox for mixture-model parameter estimation with extensive optimization capabilities. I compare two inference methods provided in this toolbox, namely:

- EM [2]:  bla
- CEM [2]: bli

You can test them with the following script: ```demo_gmm_compare.m```

#### Non-Parametric
Download the following toolboxes and make sure it's in the MATLAB path: [some other link]() and [some other link](). These toolboxes provide code for inference of the **DP-GMM** (Dirichlet Process), a realization of the Infinite Gaussian Mixture Model, which enable one to discover the number of Gaussian functions from the data, rather than doing model selection, hence the name *non-parametric inference*. I compare two inference methods for the DP-GMM, namely:

- Variational Method for DP-GMM [3]: bla
- MCMC Inference for DP-GMM [4]:     bli

You can test them with the following script: ```demo_dpgmm_compare.m```

-
#### Implementation and toolbox References:
- [1] [Gaussian Mixture Model (GMM) - Gaussian Mixture Regression (GMR)](https://www.mathworks.com/matlabcentral/fileexchange/19630-gaussian-mixture-model--gmm--gaussian-mixture-regression--gmr-) Matlab toolbox by Sylvain Calinon. 
- [2] [MixEst: An Estimation Toolbox for Mixture Models.](http://visionlab.ut.ac.ir/mixest) by Reshad Hosseini and Mohamadreza Mash'al
- [3] Some dude
