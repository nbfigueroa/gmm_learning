function [beta_star_k] = get_param_beta(beta, Nk)
% Get the beta parameter for class k
%
% beta: prior parameter on mu
% Nk: class counts
%
%


beta_star_k = beta + Nk;