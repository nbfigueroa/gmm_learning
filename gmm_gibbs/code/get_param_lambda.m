function [Lambda_star_k] = get_param_lambda(Lambda_0, beta, Nk, X_bar_k, Sk)
% Get the mu parameter for class k
%
% Lambda_0: prior parameter on wishart Lambda, D x D
% beta: prior parameter on mu
% Nk: class counts
% X_bar_k: class mean, column vector D x 1
% Sk: class covariance matrix, D x D
%

Lambda_star_k = Lambda_0 + Sk + (Nk / (beta+Nk)) * (X_bar_k*X_bar_k');