function [nu_star_k] = get_param_nu(nu, Nk)
% Get the beta parameter for class k
%
% nu: prior parameter on Lambda
% Nk: class counts
%
%


nu_star_k = nu + Nk;