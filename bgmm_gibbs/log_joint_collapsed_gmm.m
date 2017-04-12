function ll = log_joint_collapsed_gmm(data, z, Nk, X_bar, S, alpha, beta, Lambda_0, nu)
% Get the full log joint of the collapsed Gaussian mixture model; that is,
%
%   p(data, z | alpha, beta, Lambda_0, nu)
%
% where the dirichlet prior pi, and the cluster means and precisions mu_k
% and Lambda_k have been marginalized out analytically.
% 
% data: all the data, a NxD matrix
% z: an Nx1 vector of class label assignments
%
% Nk: vector Kx1, number of observations in each class
% X_bar_k: matrix KxD, mean value of observations in each class
% S: cell array Kx1, each entry a DxD empirical covariance matrix of 
%    observations within each class
% alpha: prior parameter for dirichlet pi
% beta: hyperparameter on precision of cluster means
% Lambda_0: DxD matrix hyperparameter for wishart
% nu: degrees of freedom hyperparameter for wishart


[N, D] = size(data);

K = size(Nk, 1);

ll = gammaln(K*alpha) - K*gammaln(alpha) - gammaln(K*alpha + N);

for k = 1:K
    ll = ll + gammaln(alpha+Nk(k));
end

for n = 1:N
    
    if Nk(z(n)) > 0

        mu_star_k = get_param_mu(beta,  Nk(z(n)), X_bar(z(n),:)');
        Lambda_star_k = get_param_lambda(Lambda_0, beta, Nk(z(n)), X_bar(z(n),:)', S{z(n)});
        beta_star_k = get_param_beta(beta, Nk(z(n)));
        nu_star_k = get_param_nu(nu, Nk(z(n)));

        df = nu_star_k - D + 1;
        precision = Lambda_star_k * (beta_star_k + 1) / (beta_star_k * df);

        ll = ll + logmvtpdf(data(n,:)', mu_star_k, precision, df);

    end
end