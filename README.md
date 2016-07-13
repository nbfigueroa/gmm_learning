# gmm_learning

My little toolbox for learning Gaussian Mixture Models with different inference methods in MATLAB.

-


If you are only interested in the EM method + Model Selection (BIC/AIC + CV) you do not need any external toolbox and can go directly to the demo ```gmm_em.m```.

### Inference Methods
#### Parametric:
Download the following toolbox and make sure it's in the MATLAB path: [MixEst:A MATLAB toolbox for mixture-model parameter estimation](https://github.com/utvisionlab/mixest)

- EM [1]:  ```gmm_em.m```
- CEM [2]: ```gmm_cem.m```

#### Non-Parametric
Download the following toolboxes and make sure it's in the MATLAB path: [some other link]() and [some other link]()

- Variational Method for DP-GMM [3]: ```dpgmm_vi.m```
- MCMC Inference for DP-GMM [4]:     ```dpgmm_mcmc.m```

-

#### Gaussian Mixture Regression
If you want to do regression on the learnt models, follow: ```demo_gmr.m```

-

Implementation and toolbox References:
- [1] Code provided by Sylvain Calinon. 
- [2] Reshad Hosseini and Mohamadreza Mash'al "MixEst: An Estimation Toolbox for Mixture Models." [http://visionlab.ut.ac.ir/mixest](http://visionlab.ut.ac.ir/mixest)
- [3] Some dude
