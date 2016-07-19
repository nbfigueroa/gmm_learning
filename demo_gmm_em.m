%% Testing EE Dynamics Learning Algorithms
clear all
close all
clc
%% Load Example 3D data set (positions/axis-angle)
load('./data/trajectories/example_3d.mat')

%% Convert to Trajectories to "learning-friendly" structure
pdata   = [];               % Position data
idx_r = sort(1:5,'descend');

% close all
figure('Color',[1 1 1])
for ii=1:5
    clear idx
    idx =  [1 + (5 -idx_r(ii))*length(Data)/5, length(Data) - (5-ii)*length(Data)/5];
    
    clear x
    clear o
    x = Data(1:3,idx(1):idx(2));        
    xtarget = x(:,end);
    x = x - repmat(xtarget,1,size(x,2)); % Substracting the last value from all elements of x
                                          % Now the values in x should decrease nicely to zero. Note that SEDS will
                                          % add a (0, 0, 0) to the end of each demonstration (which corresponds to
                                          % zero velocity at the end of each demo), but if it doesn't decrease nicely
                                          % it will affect the dynamics by being a too sudden change. Can cause
                                          % overshooting the attractor. It happens frequently in learning from
                                          % segmented data if not every part of the demonstration ended with zero
                                          % velocity.

    pdata{ii} = x;
   
    
    % Plot demonstration data - 3D Position    
    plot3(x(1,:),x(2,:),x(3,:),'-*', 'MarkerSize',2); hold on;
    xlabel('\xi_1')
    ylabel('\xi_2')
    zlabel('\xi_3')
    grid on; 
    axis tight;
    plot3(x(1,end),x(2,end),x(3,end), 'ok','MarkerSize',10, 'MarkerFaceColor','k'); hold on
end

%% Pre-processing for x_dot = fx(x) learning, i.e. position dynamics
dt = 0.1; %The time step of the demonstrations
tol_cutting = 0.0001; % A threshold on velocity that will be used for trimming demos
[x0 , xT, Data, index] = preprocess_demos(pdata,dt,tol_cutting); %preprocessing datas

%% Find Best Range of K to learn Position Dynamics with GMM
% Training parameters
K_range = [1:15]; %Number of Gaussian funcitons
GMM_BIC = zeros(1,length(K_range));
GMM_AIC = zeros(1,length(K_range));
GMM_LL  = zeros(1,length(K_range));

% Training of GMM by EM algorithm, initialized by k-means clustering.
for k=1:length(K_range)
    [Priors, Mu, Sigma] = EM_init_kmeans(Data, K_range(k),[]);
    [Priors, Mu, Sigma] = EM(Data, Priors, Mu, Sigma);
    w = ones(1,length(Data));
    GMM_BIC(1,k) = gmm_bic(Data,ones(1,length(Data)),Priors,Mu,Sigma,'full');
    GMM_AIC(1,k) = gmm_aic(Data,ones(1,length(Data)),Priors,Mu,Sigma,'full');
    GMM_LL(1,k)  = LogLikelihood_gmm(Data,Priors,Mu,Sigma,w);
end

%% Plot Results
figure('Color', [1 1 1])
subplot(1,2,1)
plot( K_range, GMM_BIC, '-*', 'Color', [rand rand rand]); hold on;
plot( K_range, GMM_AIC,'-*', 'Color', [rand rand rand]); hold on;
xlabel('K Gaussian Components')
legend('BIC', 'AIC')
grid on;
axis square;

subplot(1,2,2)
plot( K_range, GMM_LL, '-*', 'Color', [rand rand rand]);
xlabel('K Gaussian Components')
legend('LogLik')
grid on;
axis square;
suptitle('Model Selection for Translational Mapping Function $\hat{f_x}(\xi_x)$','interpreter','latex')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now that you found the best model you can learn a GMM representation or SEDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%% GMM learning algorithm

% Chosen K
K = 5;

% Learn Joint Distribution
fprintf('Estimating Paramaters of GMM learned through EM with %d Gaussian functions.\n', K);
tic;
[Priors_0, Mu_0, Sigma_0] = EM_init_kmeans(Data, K, []);
[Priors, Mu, Sigma] = EM(Data, Priors_0, Mu_0, Sigma_0);
toc;

pos_dyn_gmm.Priors = Priors;
pos_dyn_gmm.Mu     = Mu;
pos_dyn_gmm.Sigma  = Sigma;

%% %%%%%% SEDS learning algorithm
% A set of options that will be passed to the solver. Please type 
% 'doc preprocess_demos' in the MATLAB command window to get detailed
% information about other possible options.
clear options
options.tol_mat_bias = 10^-6; % A very small positive scalar to avoid
                              % instabilities in Gaussian kernel [default: 10^-15]
                              
options.display = 1;          % An option to control whether the algorithm
                              % displays the output of each iterations [default: true]
                              
options.tol_stopping=10^-10;  % A small positive scalar defining the stoppping
                              % tolerance for the optimization solver [default: 10^-10]

options.max_iter = 500;       % Maximum number of iteration for the solver [default: i_max=1000]

options.objective = 'likelihood';    % 'likelihood': use likelihood as criterion to
                              % optimize parameters of GMM
                              % 'mse': use mean square error as criterion to
                              % optimize parameters of GMM
                              % 'direction': minimize the angle between the
                              % estimations and demonstrations (the velocity part)
                              % to optimize parameters of GMM                              
                              % [default: 'mse']
% options.cons_penalty    = 1e2; % penalty for not going straight to the attractor. 
                                % Increase to obtain a more straight line 
K = 5;
fprintf('Estimating Paramaters of GMM learned through SEDS with %d Gaussian functions.\n', K);
tic;
[Priors_0, Mu_0, Sigma_0] = initialize_SEDS(Data,K); %finding an initial guess for GMM's parameter
[Priors Mu Sigma]=SEDS_Solver(Priors_0,Mu_0,Sigma_0,Data,options); %running SEDS optimization solver
toc;

pos_dyn_seds.Priors = Priors;
pos_dyn_seds.Mu     = Mu;
pos_dyn_seds.Sigma  = Sigma;

%% Let's See! Run Simulation for GMM Dynamics
Priors     = pos_dyn_gmm.Priors;
Mu         = pos_dyn_gmm.Mu;
Sigma      = pos_dyn_gmm.Sigma;
titlename  = 'Simulation of Translational ($\dot{\xi}=\hat{f_x}(\xi_x)$) Dynamics learned via EM';
noise      = 0;

%% Let's See! Run Simulation for SEDS Dynamics

Priors     = pos_dyn_seds.Priors;
Mu         = pos_dyn_seds.Mu;
Sigma      = pos_dyn_seds.Sigma;
titlename  = 'Simulation of Translational ($\dot{\xi}=\hat{f_x}(\xi_x)$) Dynamics learned via SEDS';
noise      = 0;

%% Run Sim
opt_sim.dt = 0.1;
opt_sim.i_max = 3000;
opt_sim.tol = 0.0005;
d = size(Data,1)/2; %dimension of data

x0_all = Data(1:d,index(1:end-1)); %finding initial points of all demonstrations

if noise
x0_all = x0_all + randn(3,5)/70; %finding initial points of all demonstrations
end

fn_handle = @(x) GMR(Priors,Mu,Sigma,x,1:d,d+1:2*d);
[x xd]=Simulation(x0_all,[],fn_handle,opt_sim); %running the simulator
hold on;
for ii=1:(length(index)-1)   
    plot3(Data(1,index(ii):index(ii+1)-1),Data(1,index(ii):index(ii+1)-1),Data(3,index(ii):index(ii+1)-1),'ob', 'MarkerSize',2); hold on;
end
view([150 11]);  
title(titlename, 'interpreter','latex')

figure('Color',[1 1 1]);
plot3(xd(1,:),xd(2,:),xd(3,:),'LineWidth',1.5); hold on; 
plot3(Data(4,:),Data(5,:),Data(6,:),'-*r', 'MarkerSize',1);
grid on;
view([150 11]);  
xlabel('$\dot{\xi_1^x}$', 'interpreter','latex')
xlabel('$\dot{\xi_2^x}$', 'interpreter','latex')
title('$\dot{\xi_3^x}$ profiles' , 'interpreter','latex')

%% Plot Learned GMM on simulation
% Plot variable interactions
figure('Color',[1 1 1], 'name', 'Learned Joint Distributions on Data')
subplot(3,2,1)
plotGMM(Mu(1:2,:), Sigma(1:2,1:2,:), [0.6 1.0 0.6], 1,[0.6 1.0 0.6]);
plot(Data(1,:),Data(2,:),'*r','MarkerSize',2); hold on;
grid on; 
xlabel('$\xi_x^1$','interpreter','latex')
ylabel('$\xi_x^2$','interpreter','latex')

subplot(3,2,3)
plotGMM(Mu([1 3],:), Sigma([1 3],[1 3],:), [0.6 1.0 0.6], 1,[0.6 1.0 0.6]);
plot(Data(1,:),Data(3,:),'*g','MarkerSize',2); hold on;
grid on; 
xlabel('$\xi_x^1$','interpreter','latex')
ylabel('$\xi_x^3$','interpreter','latex')

subplot(3,2,5)
plotGMM(Mu([2 3],:), Sigma([2 3],[2 3],:), [0.6 1.0 0.6], 1,[0.6 1.0 0.6]);
plot(Data(2,:),Data(3,:),'*b','MarkerSize',2); hold on;
grid on; 
xlabel('$\xi_x^2$','interpreter','latex')
ylabel('$\xi_x^3$','interpreter','latex')

subplot(3,2,2)
plotGMM(Mu([1 4],:), Sigma([1 4],[1 4],:), [0.6 1.0 0.6], 1,[0.6 1.0 0.6]);
plot(Data(1,:),Data(4,:),'*r','MarkerSize',2); hold on;
grid on; 
xlabel('$\xi_x^1$','interpreter','latex')
ylabel('$\dot{\xi}_x^1$','interpreter','latex')

subplot(3,2,4)
plotGMM(Mu([2 5],:), Sigma([2 5],[2 5],:), [0.6 1.0 0.6], 1,[0.6 1.0 0.6]);
plot(Data(2,:),Data(5,:),'*g','MarkerSize',2); hold on;
grid on; 
xlabel('$\xi_x^2$','interpreter','latex')
ylabel('$\dot{\xi}_x^2$','interpreter','latex')

subplot(3,2,6)
plotGMM(Mu([3 6],:), Sigma([3 6],[3 6],:), [0.6 1.0 0.6], 1,[0.6 1.0 0.6]);
plot(Data(3,:),Data(6,:),'*b','MarkerSize',2); hold on;
grid on; 
xlabel('$\xi_x^3$','interpreter','latex')
ylabel('$\dot{\xi}_x^3$','interpreter','latex')

suptitle('Learned Joint Distributions on Input/Output Data')

