# gmm_learning

My little toolbox for learning Gaussian Mixture Models with different inference methods in MATLAB.

-

If you are only interested in the standard EM (Expectation-Maximization) method [1] + Model Selection (BIC/AIC) you do not need any external toolbox and can go directly to the demo: ```simple_gmm_em.m```.

#### Gaussian Mixture Regression
If you want to do regression on the learnt models, follow: ```demo_gmr.m```. This works for any inference method.


### Other Inference Methods
#### Parametric:
Download the following toolbox and make sure it's in the MATLAB path: [MixEst](https://github.com/utvisionlab/mixest)

MixEst is a very nice MATLAB toolbox for mixture-model parameter estimation with extensive optimization capabilities.

- EM [2]:  ```gmm_em.m```
- CEM [2]: ```gmm_cem.m```

#### Non-Parametric
Download the following toolboxes and make sure it's in the MATLAB path: [some other link]() and [some other link]()

- Variational Method for DP-GMM [3]: ```dpgmm_vi.m```
- MCMC Inference for DP-GMM [4]:     ```dpgmm_mcmc.m```

-
#### Implementation and toolbox References:
- [1] [Gaussian Mixture Model (GMM) - Gaussian Mixture Regression (GMR)](https://www.mathworks.com/matlabcentral/fileexchange/19630-gaussian-mixture-model--gmm--gaussian-mixture-regression--gmr-) Matlab toolbox by Sylvain Calinon. 
- [2] Reshad Hosseini and Mohamadreza Mash'al "MixEst: An Estimation Toolbox for Mixture Models." [http://visionlab.ut.ac.ir/mixest](http://visionlab.ut.ac.ir/mixest)
- [3] Some dude
