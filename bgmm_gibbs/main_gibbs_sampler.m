% This main file is used to execute the a Gibbs sampling algorithm for 
% gaussian mixture modeling.  The data is the fisher iris data where each 
% row of data are four measurements taken from the pedal of an iris flower.  
% Important variables are listed below.
%
% data  : data matrix N x D with rows as elements of data
% z     : vector N x 1, of cluster assignments in current sample
%

clear 

load fisheriris

data = meas;
clear species meas

[N, D] = size(data);

% K is the MAXIMUM number of clusters to use
K = 15;

% we set a few hyperparameters (parameters of our prior)
alpha = 1; % dirichlet parameter
Lambda_0 = eye(D); % wishart parameter
nu = 5; % wishart degrees of freedom
beta = 1; % normal covariance parameter

% this sets the (random) initial values of the cluster assignments
z = randi(K, N, 1);

% this draws a plot of the initial labeling.
clf
figure(1)
plot_data(data,z)


%%

lp_hist = [];

% run 50 gibbs sweeps
for sweep = 1:50
    [z, lp] = run_gibbs_sweep(data, z, K, alpha, beta, Lambda_0, nu);

    figure(1);
    plot_data(data,z);
    
    lp_hist = [lp_hist lp];
    
    figure(2);
    plot(lp_hist);
    
end

%% pairs plot

figure(3);gplotmatrix(data,[], z);
