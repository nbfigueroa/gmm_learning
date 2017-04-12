function [z, lp] = run_gibbs_sweep(data, z, K, alpha, beta, Lambda_0, nu)
% Run a single Gibbs sampler sweep through the data points, resampling each
% of the N data points
%
% Returns:
%
% z: a Nx1 vector of class assignments after resampling
% lp: the log joint probability of the sampled z and the data
%
% Arguments:
%
% z: a Nx1 vector with the current class label assignments
% alpha: hyperparameter on dirichlet pi
% beta: hyperparameter on precision of cluster means
% Lambda_0: DxD matrix hyperparameter for wishart
% nu: degrees of freedom hyperparameter for wishart
%


[N, D] = size(data);


?
?
?


% loop through every data point
for n = 1:N
    data_minus_n = data(1:N~=n,:);
    x_n = data(n,:)';
    z_n_previous = z(n);
 
    ?
    ?
    
    % compute probability of each assignment
    
    ?
    ?
    ?

    % sample a new class assignment
    
    ?
    ?
end


% lp = log_joint_collapsed_gmm( ... )

end
