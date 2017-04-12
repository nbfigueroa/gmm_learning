function [mu_star_k] = get_param_mu(beta, Nk, X_bar_k)
% Get the mu parameter for class k
%
% beta: prior parameter on mu
% Nk: class counts
% X_bar_k: class mean, column vector D x 1
%
%


mu_star_k = Nk * X_bar_k / (beta + Nk);