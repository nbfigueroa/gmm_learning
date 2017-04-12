function [lp] = log_predictive_collapsed_gmm(x, K, N, Nk, X_bar_k, Sk, alpha, beta, Lambda_0, nu)
% Compute the (unnormalized) log predictive for a single class; that is,
% 
%   p(x_new=k | data, alpha, beta, Lambda_0, nu)
%
% where the dirichlet prior pi, and the cluster means and precisions mu_k
% and Lambda_k have been marginalized out analytically.
%
% x: new data point, D x 1 vector
% K: (maximum) number of classes
% N: total number of observations
% Nk: number of observations within this class (not including x)
% X_bar_k: mean value of observations within this class (not including x)
% Sk: empirical covariance matrix of observations within this class (not including x)
% alpha: prior parameter for dirichlet pi
% beta: hyperparameter on precision of cluster means
% Lambda_0: DxD matrix hyperparameter for wishart
% nu: degrees of freedom hyperparameter for wishart


D = size(Sk, 1);
assert(size(X_bar_k,1) == D);

% compute contribution from dirichlet-multinomial
log_dirichlet = log(alpha + Nk) - log(K*alpha + N);

% compute parameters for student t
mu_star_k = get_param_mu(beta,  Nk, X_bar_k);
Lambda_star_k = get_param_lambda(Lambda_0, beta, Nk, X_bar_k, Sk);
beta_star_k = get_param_beta(beta, Nk);
nu_star_k = get_param_nu(nu, Nk);

df = nu_star_k - D + 1;
W = Lambda_star_k * (beta_star_k + 1) / (beta_star_k * df);

log_mvt = logmvtpdf(x, mu_star_k, W, df);

lp = log_dirichlet + log_mvt;
