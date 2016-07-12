# gmm_learning

My little toolbox for learning Gaussian Mixture Models with different inference methods in MATLAB.

-

If you are only interested in the EM method + Model Selection (BIC/AIC + CV) you do not need any external toolbox and can go directly to the demo ```gmm_em.m```.

### Inference Methods
#### Parametric:
Download the following toolbox and make sure it's in the MATLAB path: [some link]()

- EM [1]:  ```demo_gmm_em.m```
- CEM [2]: ```demo_gmm_cem.m```

#### Non-Parametric
Download the following toolboxes and make sure it's in the MATLAB path: [some other link]() and [some other link]()

- Variational Method for DP-GMM [3]: ```demo_dpgmm_vi.m```
- MCMC Inference for DP-GMM [4]:     ```demo_dpgmm_mcmc.m```

#### Gaussian Mixture Regression
If you want to do regression on the learnt models, follow: ```demo_gmr.m```
