%% Testing EE Dynamics Learning Algorithms
clear all
close all
clc
%% Load Example 3D data set (positions/axis-angle)
load('example_3d.mat')
change_ori = 1;

%% Convert to SEDS-friendly data structures
pdata   = [];               % Position data
odata   = [];               % Orientation data
otarget = [];   % Target in orientation (The target for position will be 0,0,0)
idx_r = sort(1:5,'descend');

% close all
figure('Color',[1 1 1])
for ii=1:5
    clear idx
    idx =  [1 + (5 -idx_r(ii))*length(Data)/5, length(Data) - (5-ii)*length(Data)/5];
    
    clear x
    clear o
    x = Data(1:3,idx(1):idx(2));
    o = Data(4:7,idx(1):idx(2));

    otarget = [otarget, o(:,end)];      % Setting the target of the orientation as the last point of the demonstrations
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
    plot3(x(1,end),x(2,end),x(3,end), 'ok','MarkerSize',10, 'MarkerFaceColor','k'); hold on;
    
    clear q
    q = zeros (size(o));
    for kk=1:size(q,2), q(:,kk) = AxisAngle2Quat(o(1:3,kk),o(4,kk));end
    
    noise = randn(2,1)/4;    
    if change_ori        
        R = quaternion(q);
        R(:,:,1) = rotx(pi + noise(1))*rotz(-pi/4)*roty(noise(2));
        q(:,1) = quaternion(R(:,:,1));
        qt = [0;0;0;1];
        for jj=2:size(o,2)
            slerp_t = exp(-25*norm(x(:,jj)));
            q(:,jj) = slerp(q(:,jj-1),qt,slerp_t);
        end
        R = quaternion(q);
        
       for kk=1:size(o,2) 
        [ w,phi ] = R2AxisAngle(R(:,:,kk));
        o(1:3,kk) = w; o(4,kk) = phi;
        end
    end
    

    
    odata{ii} = o;    
    % Convert to Homogeneous MatriX Representation
    clear H
    X = [x;q]; H = convert2H(X);
    for jj=1:3:size(q,2),
        % Draw Frame
        drawframe(H(:,:,jj),0.005);
        % Draw EE
        [xyz] = R2rpy(H(1:3,1:3,jj));
        xyz = xyz*180/pi;
        t = x(:,jj);
%         drawLine3d([t(1) t(2) t(3) H(1,3,jj)/10 H(2,3,jj)/10 H(3,3,jj)/10])
        drawCuboid([t(1) t(2) t(3) 0.005 0.005 0.01 xyz(3) xyz(2) xyz(1)], 'FaceColor', 'g');
        alpha(.5)
    end
    hold on;
    view([-66 11]);
    title('Position/Orientation Data \xi = [x,y,z,s_x,s_y,s_z,\theta]')
end


otarget_ = mean(otarget,2);
Htarget = convert2H([xtarget;AxisAngle2Quat(otarget(1:3),otarget(4))]);
drawframe(Htarget,0.02)


%% Pre-processing for SEDS fx learning, i.e. position dynamics
dt = 0.1; %The time step of the demonstrations
tol_cutting = 0.0001; % A threshold on velocity that will be used for trimming demos
[x0 , xT, Data, index] = preprocess_demos(pdata,dt,tol_cutting); %preprocessing datas

% Check the pre-processing
plot3DDynamics(Data, index, 'Position', 'Velocity');

%% Check QQ-PLot of Normality
pos = Data(1:3,:);
vel = Data(4:6,:);

%%
x = sort(pos(:));
GMModel = fitgmdist(x,6);
np= 1000;
p= (0+1/(np-1):1/(np-1):1-1/(np-1))';

for j=1:GMModel.NumComponents
     y = y + GMModel.ComponentProportion(j)*norminv(p,GMModel.mu(j,:),GMModel.Sigma(:,:,j));
end

figure
hold on
qd= quantile(x,p);
plot(y,qd,'+b');
plot([min(y) max(y)],[min(y) max(y)],'r-.');
xlabel('Theoretical Quantiles'); 
ylabel('Quantiles of Input Sample');
title('Gaussian Mixture')

%%
figure('Color',[1 1 1]);
subplot(3,2,1)
gqqplot(pos(:),'uniform')

subplot(3,2,2)
gqqplot(vel(:),'uniform')

subplot(3,2,3)
gqqplot(pos(:),'normal')

subplot(3,2,4)
gqqplot(vel(:),'normal')

subplot(3,2,5)
hist(pos(:));hold on
% x = [min(pos(:)):max(pos(:))];
% norm = normpdf(x,GMModel.mu(1),GMModel.Sigma(:,:,1));
% plot(x,norm)

subplot(3,2,6)
hist(vel(:))

suptitle('Position(left)/ Velocity(right) Variable Tests')
%% Find Best Range of K to learn SEDS Model with GMM
% Training parameters
K_range = [1:15]; %Number of Gaussian funcitons
GMM_BIC = zeros(1,length(K_range));
GMM_AIC = zeros(1,length(K_range));
GMM_LL  = zeros(1,length(K_range));

% Training of GMM by EM algorithm, initialized by k-means clustering.
for k=1:length(K_range)
    [Priors, Mu, Sigma] = EM_init_kmeans(Data, K_range(k));
    [Priors, Mu, Sigma] = EM(Data, Priors, Mu, Sigma);
    w = ones(1,length(Data));
    GMM_BIC(1,k) = gmm_bic(Data,ones(1,length(Data)),Priors,Mu,Sigma,'full');
    GMM_AIC(1,k) = gmm_aic(Data,ones(1,length(Data)),Priors,Mu,Sigma,'full');
    GMM_LL(1,k)  = LogLikelihood_gmm(Data,Priors,Mu,Sigma,w);
end

%%
figure('Color', [1 1 1])
plot( K_range, GMM_BIC, '-*', 'Color', [rand rand rand]); hold on;
plot( K_range, GMM_AIC,'-*', 'Color', [rand rand rand]); hold on;
xlabel('K Gaussian Components')
legend('BIC', 'AIC')
title('Model Selection for Translational Mapping Function $\hat{f_x}(\xi_x)$','interpreter','latex')
grid on;

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
%% Putting GMR and SEDS library in the MATLAB Path
if isempty(regexp(path,['SEDS_lib' pathsep], 'once'))
    addpath([pwd, '/seds/SEDS_lib']);    % add SEDS dir to path
end
if isempty(regexp(path,['GMR_lib' pathsep], 'once'))
    addpath([pwd, '/seds/GMR_lib']);    % add GMR dir to path
end
%% SEDS learning algorithm
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

options.max_iter = 1000;       % Maximum number of iteration for the solver [default: i_max=1000]

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
K = 6;
fprintf('Estimating Paramaters of GMM learned through SEDS with %d Gaussian functions.\n', K);
tic;
[Priors_0, Mu_0, Sigma_0] = initialize_SEDS(Data,K); %finding an initial guess for GMM's parameter
[Priors Mu Sigma]=SEDS_Solver(Priors_0,Mu_0,Sigma_0,Data,options); %running SEDS optimization solver
toc;

pos_dyn_seds.Priors = Priors;
pos_dyn_seds.Mu     = Mu;
pos_dyn_seds.Sigma  = Sigma;

fprintf('Estimating Paramaters of GMM learned through EM with %d Gaussian functions.\n', K);
tic;
[Priors_0, Mu_0, Sigma_0] = EM_init_kmeans(Data, K);
[Priors, Mu, Sigma] = EM(Data, Priors_0, Mu_0, Sigma_0);
toc;

pos_dyn_gmm.Priors = Priors;
pos_dyn_gmm.Mu     = Mu;
pos_dyn_gmm.Sigma  = Sigma;


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



%% Pre-processing for SEDS fo learning, i.e. orientation dynamics
dt = 0.1; %The time step of the demonstrations
tol_cutting = 0.0000001; % A threshold on velocity that will be used for trimming demos
% close all

% Convert odata to Scaled-AxisAngle like Sengsue (supossedly) or Quaternion
odata_sc = [];
for ii=1:length(odata)    
    
    o = odata{ii};
    % Remove peaks
    for jj=1:size(o,1), o(jj,:) = medfilt1(o(jj,:),5);end
    
    % Decompose Axis and Angle
    clear axes angle norm
    axes = o(1:3,:);
    angle = o(4,:);
    tmp = zeros(3,length(axes));

    % Scaled Axis-Angle
    for jj=1:length(axes) 
        tmp(:,jj) =  axes(:,jj)/norm(axes(:,jj))*angle(jj); 
    end       
    
    % Quaternion
    clear q
    q = zeros (size(o));
    for kk=1:size(q,2), q(:,kk) = AxisAngle2Quat(o(1:3,kk),o(4,kk)); ...
    q(:,kk) = q(:,kk)/norm(q(:,kk));end
    
    odata_sc{ii} = q;    
end

%% Preprocessing data for SEDS
[x0 , xT, Data, index] = preprocess_demos(odata_sc,dt,tol_cutting); 

% Shift by Pi on Z
% Data = Data - repmat(Data(:,end),[1 size(Data,2)]);

% Check the pre-processing
plot3DDynamics(Data, index, 'Scaled Axis-Angle', 'Scaled Axis-Angle Time Rates');

%% Check QQ-PLot of Normality
ang = Data(1:3,:);
ang_diff = Data(4:6,:);

figure('Color',[1 1 1]);
subplot(3,2,1)
gqqplot(ang(:),'uniform')

subplot(3,2,2)
gqqplot(ang_diff(:),'uniform')

subplot(3,2,3)
gqqplot(ang(:),'normal')

subplot(3,2,4)
gqqplot(ang_diff(:),'normal')

subplot(3,2,5)
hist(ang(:))

subplot(3,2,6)
hist(ang_diff(:))

suptitle('Angle (left)/ Angular Time Rates(right) Variable Tests')

figure('Color',[1 1 1]);

subplot(2,3,1)
circ_plot(Data(1,:)','pretty','bo',true,'linewidth',2,'color','r'),

title('x-angle')
subplot(2,3,4)
circ_plot(Data(1,:)','hist',[],20,true,true,'linewidth',2,'color','r')



subplot(2,3,2)
circ_plot(Data(2,:)','pretty','bo',true,'linewidth',2,'color','r'),
title('y-angle')

subplot(2,3,5)
circ_plot(Data(2,:)','hist',[],20,true,true,'linewidth',2,'color','r')


subplot(2,3,3)
circ_plot(Data(3,:)','pretty','bo',true,'linewidth',2,'color','r'),
title('z-angle')

subplot(2,3,6)
circ_plot(Data(3,:)','hist',[],20,true,true,'linewidth',2,'color','r')



%% Find Best Range of K to learn SEDS Model with GMM
% Training parameters
K_range = [1:30]; %Number of Gaussian funcitons
GMM_BIC = zeros(1,length(K_range));
GMM_AIC = zeros(1,length(K_range));
GMM_LL  = zeros(1,length(K_range));

% Training of GMM by EM algorithm, initialized by k-means clustering.
for k=1:length(K_range)
    [Priors, Mu, Sigma] = EM_init_kmeans(Data, K_range(k));
    [Priors, Mu, Sigma] = EM(Data, Priors, Mu, Sigma);
    w = ones(1,length(Data));
    GMM_BIC(1,k) = gmm_bic(Data,w,Priors,Mu,Sigma,'full');
    GMM_AIC(1,k) = gmm_aic(Data,w,Priors,Mu,Sigma,'full');
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
% suptitle('Model Selection for Rotational Mapping Function')
%% SEDS learning algorithm
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

options.objective = 'mse';    % 'likelihood': use likelihood as criterion to
                              % optimize parameters of GMM
                              % 'mse': use mean square error as criterion to
                              % optimize parameters of GMM
                              % 'direction': minimize the angle between the
                              % estimations and demonstrations (the velocity part)
                              % to optimize parameters of GMM                              
                              % [default: 'mse']
K = 50;
fprintf('Estimating Paramaters of GMM learned through SEDS with %d Gaussian functions.\n', K);
tic;
[Priors_0, Mu_0, Sigma_0] = initialize_SEDS(Data,K); %finding an initial guess for GMM's parameter
[Priors Mu Sigma]=SEDS_Solver(Priors_0,Mu_0,Sigma_0,Data,options); %running SEDS optimization solver
toc;

ori_dyn_seds.Priors = Priors;
ori_dyn_seds.Mu     = Mu;
ori_dyn_seds.Sigma  = Sigma;
%%
K = 50;
fprintf('Estimating Paramaters of GMM learned through EM with %d Gaussian functions.\n', K);
tic;
[Priors_0, Mu_0, Sigma_0] = EM_init_kmeans(Data, K);
[Priors, Mu, Sigma] = EM(Data, Priors_0, Mu_0, Sigma_0);
toc;

ori_dyn_gmm.Priors = Priors;
ori_dyn_gmm.Mu     = Mu;
ori_dyn_gmm.Sigma  = Sigma;

%% Let's See! Run Simulation for GMM Dynamics

clear Prios Mu Sigma
Priors     = ori_dyn_gmm.Priors;
Mu         = ori_dyn_gmm.Mu;
Sigma      = ori_dyn_gmm.Sigma;
titlename  = 'Ori Dynamics with GMM';

%% Let's See! Run Simulation for SEDS Dynamics

clear Prios Mu Sigma
Priors     = ori_dyn_seds.Priors;
Mu         = ori_dyn_seds.Mu;
Sigma      = ori_dyn_seds.Sigma;
titlename  = 'Ori Dynamics with SEDS';

%% Let's See! Run Simulation for GMM Dynamics learnt from MixEst
K = 6;
gmm_mixest_dyn = GMM_mdls{K}.theta;
clear Prios Mu Sigma
Priors = gmm_mixest_dyn.p';
Mu = zeros(size(Data,1),K); Sigma = zeros(size(Data,1),size(Data,1),K); 
for ii=1:K, Mu(:,ii) = gmm_mixest_dyn.D{ii}.mu; ...
Sigma(:,:,ii) = gmm_mixest_dyn.D{ii}.sigma;end
titlename  = 'Ori Dynamics with GMM (MixEst)';

%% Run Sim
opt_sim.dt = 0.1;
opt_sim.i_max = 3000;
opt_sim.tol = 0.0005;
d = size(Data,1)/2; %dimension of data

x0_all = Data(1:d,index(1:end-1)) + repmat(0,size(x0_all)); %finding initial points of all demonstrations
fn_handle = @(x) GMR(Priors,Mu,Sigma,x,1:d,d+1:2*d);
[x_r xd_r]=Simulation(x0_all,[],fn_handle,opt_sim); %running the simulator
hold on;
for ii=1:(length(index)-1)   
    plot3(Data(1,index(ii):index(ii+1)-1),Data(2,index(ii):index(ii+1)-1),Data(3,index(ii):index(ii+1)-1),'*b', 'MarkerSize',1); hold on;
end
view([150 11]);  
title(titlename)

figure('Color',[1 1 1]);
plot3(xd_r(1,:),xd_r(2,:),xd_r(3,:),'LineWidth',1.5); hold on; 
plot3(Data(4,:),Data(5,:),Data(6,:),'-*r', 'MarkerSize',1);
grid on;
view([150 11]);  
xlabel('$\dot{\xi_1^r}$', 'interpreter','latex')
xlabel('$\dot{\xi_2^r}$', 'interpreter','latex')
title('$\dot{\xi_3^r}$ profiles' , 'interpreter','latex')


%% Plot Learned GMM on simulation
% Plot variable interactions
figure('Color',[1 1 1], 'name', 'Learned Joint Distributions on Data')
subplot(3,2,1)
plotGMM(Mu(1:2,:), Sigma(1:2,1:2,:), [0.6 1.0 0.6], 1,[0.6 1.0 0.6]);
plot(Data(1,:),Data(2,:),'*r','MarkerSize',2); hold on;
grid on; 
xlabel('$\xi_r^1$','interpreter','latex')
ylabel('$\xi_r^2$','interpreter','latex')

subplot(3,2,3)
plotGMM(Mu([1 3],:), Sigma([1 3],[1 3],:), [0.6 1.0 0.6], 1,[0.6 1.0 0.6]);
plot(Data(1,:),Data(3,:),'*g','MarkerSize',2); hold on;
grid on; 
xlabel('$\xi_r^1$','interpreter','latex')
ylabel('$\xi_r^3$','interpreter','latex')

subplot(3,2,5)
plotGMM(Mu([2 3],:), Sigma([2 3],[2 3],:), [0.6 1.0 0.6], 1,[0.6 1.0 0.6]);
plot(Data(2,:),Data(3,:),'*b','MarkerSize',2); hold on;
grid on; 
xlabel('$\xi_r^2$','interpreter','latex')
ylabel('$\xi_r^3$','interpreter','latex')

subplot(3,2,2)
plotGMM(Mu([1 4],:), Sigma([1 4],[1 4],:), [0.6 1.0 0.6], 1,[0.6 1.0 0.6]);
plot(Data(1,:),Data(4,:),'*r','MarkerSize',2); hold on;
grid on; 
xlabel('$\xi_r^1$','interpreter','latex')
ylabel('$\dot{\xi}_r^1$','interpreter','latex')

subplot(3,2,4)
plotGMM(Mu([2 5],:), Sigma([2 5],[2 5],:), [0.6 1.0 0.6], 1,[0.6 1.0 0.6]);
plot(Data(2,:),Data(5,:),'*g','MarkerSize',2); hold on;
grid on; 
xlabel('$\xi_r^2$','interpreter','latex')
ylabel('$\dot{\xi}_r^2$','interpreter','latex')

subplot(3,2,6)
plotGMM(Mu([3 6],:), Sigma([3 6],[3 6],:), [0.6 1.0 0.6], 1,[0.6 1.0 0.6]);
plot(Data(3,:),Data(6,:),'*b','MarkerSize',2); hold on;
grid on; 
xlabel('$\xi_r^3$','interpreter','latex')
ylabel('$\dot{\xi}_r^3$','interpreter','latex')

suptitle('Learned Joint Distributions on Input/Output Data')


%% Full 6D simulation

clear q            
t = x(:,1:100);
o = x_r(:,1:100);
q = zeros (4,length(o));
for kk=1:size(o,2), 
    angle = norm(o(:,jj));
    axis = o(:,j)*angle;   
    q(:,kk) = AxisAngle2Quat(axis,angle);
end

% Convert to Homogeneous MatriX Representation
clear H
X = [t;q]; H = convert2H(X);
for jj=1:5:size(q,2)
    % Draw FramePosition/Orientation Data \xi = [x,y,z,s_x,s_y,s_z,\theta]
    drawframe(H(:,:,jj),0.005);
    % Draw EE
    [xyz] = R2rpy(H(1:3,1:3,jj));
    xyz = xyz*180/pi;
    t = x(:,jj);
    drawCuboid([t(1) t(2) t(3) 0.05 0.05 0.01 xyz(3) xyz(2) xyz(1)], 'FaceColor', 'g');
    alpha(.5)
end
grid on;
hold on;
view([-66 11]);
title('Simulate 6D Motion')
    
%% GMM Learning with EM on orientation dynamics

% main options
clear options
options.verbosity = 2;
options.solver = 'default';
options.tolcostdiff = 1e-3;
options.maxIter = 500;
options.penalize = true;
options.regularize = true;

K_range  = 1:50;
GMM2_BIC  = zeros(1,length(K_range));
GMM2_AIC  = zeros(1,length(K_range));
GMM2_LL   = zeros(1,length(K_range));
GMM_mdls = {};

data = Data(1:4,:);
dim = size(data,1);
for k=1:length(K_range)    
    D = mixturefactory(mvnfactory(dim), K_range(k));

    % perform estimation
    theta    = D.estimate(data, options);
    
    % Gather Stats
    GMM2_LL(1,k)   = D.ll(theta,data);    
    [GMM2_AIC(1,k), GMM2_BIC(1,k)]  = gmm_ic_metrics(data, K_range(k), D.ll(theta,data), 'full');
    
    % Gather Models
    GMM_mdls{k}.theta = theta;
    GMM_mdls{k}.D     = D;
    
end

%% Plot Results
figure('Color', [1 1 1])
subplot(1,2,1)
plot( K_range, GMM2_BIC, '-*', 'Color', [rand rand rand]); hold on;
plot( K_range, GMM2_AIC, '-*', 'Color', [rand rand rand]); hold on;
xlabel('K Gaussian Components')
legend('BIC','AIC')
grid on;
axis square;

subplot(1,2,2)
plot( K_range, GMM2_LL,'-*', 'Color', [rand rand rand]); 
xlabel('K Gaussian Components')
ylabel('LogLik')
grid on;
axis square;

%% moVMF Learning with EM on orientation dynamics

% main options
clear options
options.verbosity = 2;
options.solver = 'default';
options.tolcostdiff = 1e-5;
options.maxIter = 500;
% options.penalize = true;


K_range  = 1:100;
moVMF_BIC  = zeros(1,length(K_range));
moVMF_AIC  = zeros(1,length(K_range));
moVMF_LL   = zeros(1,length(K_range));
moVMF_mdls = {};

check_norm = 1;
data = Data(1:4,:);
dim = size(data,1);

% Check Norm of Data Vectors
if check_norm
    norms = [];
    for ii=1:(size(data,2)-1),
        data(:,ii) = data(:,ii)/norm(data(:,ii));
        norms = [norms; norm(data(:,ii))];
    end        
end

for k=1:length(K_range)    
    D = mixturefactory(vmffactory(dim), K_range(k));

    % perform estimation
    theta    = D.estimate(data, options);        
    
    % Gather Stats
    moVMF_LL(1,k)   = D.ll(theta,data);
    [moVMF_AIC(1,k), moVMF_BIC(1,k)]  = moVMF_ic_metrics(data, K_range(k), D.ll(theta,data));
    
    % Gather Models
    moVMF_mdls{k}.theta = theta;
    moVMF_mdls{k}.D     = D;
    
end

%% Plot Results
figure('Color', [1 1 1])
subplot(1,2,1)
plot( K_range, moVMF_BIC, '-*', 'Color', [rand rand rand]); hold on;
plot( K_range, moVMF_AIC, '-*', 'Color', [rand rand rand]); hold on;
xlabel('K von Mises-Fisher Components')
legend('BIC','AIC')
grid on;
axis square;

subplot(1,2,2)
plot( K_range, moVMF_LL,'-*', 'Color', [rand rand rand]); 
xlabel('K von Mises-Fisher Components')
ylabel('LogLik')
grid on;
axis square;
