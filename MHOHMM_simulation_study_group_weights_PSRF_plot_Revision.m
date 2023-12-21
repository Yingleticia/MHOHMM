%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --              Mixed Higher Order Hidden Markov Model              -- %%%
%%% --                       Simulation study                           -- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% -- By Ying Liao, Last modified in December, 2021 -- %

% -- Download and Install Tensor Toolbox for Matlab -- %
% -- Installation Instructions -- %
% 1. Unpack the files.
% 2. Rename the root directory of the toolbox from tensor_toolbox_2.5 to tensor_toolbox.
% 3. Start MATLAB.
% 4. Within MATLAB, cd to the tensor_toolbox directory and execute the following commands. 
%       addpath(pwd)            %<-- add the tensor toolbox to the MATLAB path
%       cd met; addpath(pwd)    %<-- also add the met directory
%       savepath                %<-- save for future MATLAB sessions
clear;clc;close all;

dataGenerate = 0; % 1: generate new MCMC samples; 0: using saved data

%% Set seed 
if dataGenerate == 0
    
    simsize = 2000; % 5000; 2000
    if simsize == 2000
        burnin = 1000; 
        gap = 1; % 1
    elseif simsize == 5000
        burnin = 3000; 
        gap = 10;
    end
    if gap == 1
        filenameSave = ['PSRF_plot_data_no_gap_',num2str(simsize/1000),'k'];
    else
        filenameSave = ['PSRF_plot_data_',num2str(simsize/1000),'k']; 
    end
else
    level_N = 2; % number of groups: 2,4,6,8,10 (sensitivity analysis)
    dmax = 3; % 3,4,5,6,7 (sensitivity analysis)
    qmax = 5; % 3,4,5 (sensitivity analysis)
    folder = ['Result_AOAS/Classification/Revision/Groups_',num2str(level_N),'/'];

    seed_opt = 12; % CV

    seed = 1; % replicates

    chain_num = 6;


    % dbstop if error;
    rng(seed);
    RandStream.getGlobalStream;

    %%%%%%%% save samples for PSRF plots %%%%%%%% 
    %------ 2 models (N; NA); MCMC chains ------% 
    comp_time = nan(2,chain_num);
    %-----> Observed process
    mu_storage_all=cell(2,chain_num);
    beta_storage_all=cell(2,chain_num);
    sigma_storage_all=cell(2,chain_num);
    sigma_r_storage_all=cell(2,chain_num);

    %-----> Hidden process
    lambda_x0_storage_all=cell(2,chain_num);
    omega_storage_all=cell(2,chain_num);
    alpha_x0_storage_all=cell(2,chain_num);
    alpha_x1_storage_all=cell(2,chain_num);

    %% load data and parameters
    filename0=[folder,'Final_data_Seed_',num2str(seed_opt),'_q_',num2str(qmax),'_d_',num2str(dmax),'.mat'];
    load(filename0,'model_str_best','idx_N','Train_Num_N','T','dmax','qmax','N1',...
            'level_N','level_Train_N','Y_Train_N','Y_Train_total_N','T_N','alpha_x',...
            'alpha00','Parameters_N','likelihood_type','Parameters_prior_N','pM',...
            'simsize','burnin','gap','pigamma','alpha_x0','alpha_x1','alpha00',...
            'alpha_a0','alpha_b0','alpha_a1','alpha_b1','omega','omega_a','omega_b',...
            'idx_NA','Train_Num_NA','level_Train_NA','Y_Train_NA','Y_Train_total_NA',...
            'T_NA','Parameters_NA','Parameters_prior_NA','level_Test_N',...
            'level_Test_NA','Y_Test_N','Y_Test_NA','Test_Num_N','Test_Num_NA')  

    %%% change the number of iterations %%%
    if 1 == 0
        simsize = 5000;
        burnin = 3000;
        gap = 10; 
    end    

    %% Multiple MCMC chains
    for ch = 1:chain_num
    tic
    %%%%%%%%%%%%%%%
    %%% Model N %%%
    %%%%%%%%%%%%%%%

    beta=0.5; % initial fixed effect
    C_total=idx_N;
    CC_K=cell(1,Train_Num_N); % initial state sequences 
    for aa=1:Train_Num_N
        CC_K{aa}=C_total((T*(aa-1)+1):(T*aa));
    end

    %%% Sampler for Stage 1
    fprintf('\n--- MHOHMM (Model N) --- \n\n');
    fprintf('\n--- First Stage --- \n\n');

    [Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
        lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
        Train_Num_N,level_N,level_Train_N,CC_K,Y_Train_N,...
        Y_Train_total_N,T_N,alpha_x,alpha00,...
        Parameters_N,beta,likelihood_type,Parameters_prior_N,pM,model_str_best);


    %%% Sampler for Stage 2
    fprintf('\n --- Second Stage --- \n\n');
    pause(1);
    [lambda_x_Mix_N,lambda_x0_Mix_N,alpha_x0_Mix_N,alpha_x1_Mix_N,omega_Mix_N,...
        pi_Mix_N,mu_Mix_N,sigma_Mix_N,beta_Mix_N,Sigma_r_Mix_N,L00_Mix_N,...
        State_select_Mix_N,p00_Mix_N,K00_Mix_N,ind00_Mix_N,likeli_Mix_N]=...
        MHOHMM_str_stage2_aoas_diagnostic(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Train_Num_N,...
        level_N,level_Train_N,C_total,CC_K,Xnew,Cnew,Y_Train_N,Y_Train_total_N,...
        T_N,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,....
        alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
        Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_N,model_str_best);

    comp_time(1,ch) = toc/60/60;
    %-----> Observed process
    mu_storage_all{1,ch} = mu_Mix_N;
    beta_storage_all{1,ch} = beta_Mix_N;
    sigma_storage_all{1,ch} = sigma_Mix_N;
    sigma_r_storage_all{1,ch} = Sigma_r_Mix_N;

    %-----> Hidden process
    lambda_x0_storage_all{1,ch} = lambda_x0_Mix_N;
    omega_storage_all{1,ch} = omega_Mix_N;
    alpha_x0_storage_all{1,ch} = alpha_x0_Mix_N;
    alpha_x1_storage_all{1,ch} = alpha_x1_Mix_N;

    %%
    tic
    %%%%%%%%%%%%%%%%
    %%% Model NA %%%
    %%%%%%%%%%%%%%%%

    beta=0.5; % initial fixed effect
    C_total=idx_NA;
    CC_K=cell(1,Train_Num_NA); % initial state sequences 
    for aa=1:Train_Num_NA
        CC_K{aa}=C_total((T*(aa-1)+1):(T*aa));
    end

    %%% Sampler for Stage 1
    fprintf('\n--- MHOHMM (Model NA) --- \n\n');
    fprintf('\n--- First Stage --- \n\n'); 

    [Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
        lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
        Train_Num_NA,level_N,level_Train_NA,CC_K,Y_Train_NA,...
        Y_Train_total_NA,T_NA,alpha_x,alpha00,Parameters_NA,beta,...
        likelihood_type,Parameters_prior_NA,pM,model_str_best);


    %%% Sampler for Stage 2
    fprintf('\n --- Second Stage --- \n\n');
    pause(1);
    [lambda_x_Mix_NA,lambda_x0_Mix_NA,alpha_x0_Mix_NA,alpha_x1_Mix_NA,omega_Mix_NA,pi_Mix_NA,...
        mu_Mix_NA,sigma_Mix_NA,beta_Mix_NA,Sigma_r_Mix_NA,L00_Mix_NA,...
        State_select_Mix_NA,p00_Mix_NA,K00_Mix_NA,ind00_Mix_NA,likeli_Mix_NA]=...
        MHOHMM_str_stage2_aoas_diagnostic(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Train_Num_NA,...
        level_N,level_Train_NA,C_total,CC_K,Xnew,Cnew,Y_Train_NA,Y_Train_total_NA,...
        T_NA,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,...
        alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
        Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_NA,model_str_best);

    comp_time(2,ch) = toc/60/60;
    %-----> Observed process
    mu_storage_all{2,ch} = mu_Mix_NA;
    beta_storage_all{2,ch} = beta_Mix_NA;
    sigma_storage_all{2,ch} = sigma_Mix_NA;
    sigma_r_storage_all{2,ch} = Sigma_r_Mix_NA;

    %-----> Hidden process
    lambda_x0_storage_all{2,ch} = lambda_x0_Mix_NA;
    omega_storage_all{2,ch} = omega_Mix_NA;
    alpha_x0_storage_all{2,ch} = alpha_x0_Mix_NA;
    alpha_x1_storage_all{2,ch} = alpha_x1_Mix_NA;


    end
    disp('========== Computation Time (in hours) ==========') 
    Model={'N';'NA'};
    temp=table(Model);
    Chain=array2table(comp_time);
    disp([temp,Chain])

    %%% save data for PSRF plots %%%
    filename = ['PSRF_plot_data_no_gap_',num2str(simsize/1000),'k'];
    save(filename,'mu_storage_all','beta_storage_all',...
        'sigma_storage_all','sigma_r_storage_all','lambda_x0_storage_all',...
        'omega_storage_all','alpha_x0_storage_all','alpha_x1_storage_all')
end



%% Convergence diagnostics
if dataGenerate == 0
    load(filenameSave)
    chain_num = size(mu_storage_all,2);
end

fig = 1;

model_select = 1; % 1: N; 2: NA
chain_select = [2,5]; % range: {1,...,chain_num} [4,5]; no-gap: [2,5]
point_start = 5;

color_order = rand(30,3);
Ymax = 0; % 0: no setting
lineSZ = 1;
fontSZ = 20;
FigOne = 2; % 1: plot in the same figure for all parameters; 2: for two processes
lgd = 0;% legend or not
if lgd == 0
    FigOne = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%  PSRF computation  %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%---- state ordering for mu; sigma; lambda_0_x ----%
mu_chain = mu_storage_all(model_select,:);
sigma_chain = sigma_storage_all(model_select,:);
lambda_x0_chain = lambda_x0_storage_all(model_select,:);

beta_chain = beta_storage_all(model_select,:);
sigma_r_chain = sigma_r_storage_all(model_select,:);
omega_chain = omega_storage_all(model_select,:);
alpha_x0_chain = alpha_x0_storage_all(model_select,:);
alpha_x1_chain = alpha_x1_storage_all(model_select,:);


if 1 == 1 % sort or not
    
    [dmax, level_num, B] = size(lambda_x0_chain{1}); 
    % dmax: # of states; B: # of samples; level_num: # of groups
    order_idx = repmat((1:dmax)',1,chain_num);
    
    for aa=1:chain_num
        temp = mean(mu_chain{aa},2);
        [~,I] = sort(temp);
        order_idx(:,aa) = I;
    end 
    
    for aa=1:chain_num
        I = order_idx(:,aa);
        %-- mu --%
        temp = mu_chain{aa};
        temp2 = temp(I,:);
        mu_chain{aa} = temp2;
        %-- sigma --%
        temp = sigma_chain{aa};
        temp2 = temp(I,:);
        sigma_chain{aa} = temp2;
        %-- sigma --%
        temp = lambda_x0_chain{aa};
        temp2 = temp(I,:,:);
        lambda_x0_chain{aa} = temp2;
    end
    
end


%%%----------------------------  mu  ----------------------------%%%
X = zeros(B,dmax,chain_num);
for ch = 1:chain_num
    temp = mu_chain{ch};
    for d = 1:dmax
        X(:,d,ch) = temp(d,:); % state d
    end
end
X = X(:,:,chain_select);
R_mu = zeros(dmax,B-point_start);
for aa = 1:(B-point_start)
    for d = 1:dmax
        [R,~,~,~,~] = psrf(X(1:(aa+point_start),d,:));
        R_mu(d,aa)=R;
    end   
end

%%%----------------------------  sigma  ----------------------------%%%
X = zeros(B,dmax,chain_num);
for ch = 1:chain_num
    temp = sigma_chain{ch};
    for d = 1:dmax
        X(:,d,ch) = temp(d,:); % state d
    end
end
X = X(:,:,chain_select);
R_sigma = zeros(d,B-point_start);
for aa = 1:(B-point_start)
    for d = 1:dmax
        [R,~,~,~,~] = psrf(X(1:(aa+point_start),d,:));
        R_sigma(d,aa)=R;
    end   
end

%%%----------------------------  beta  ----------------------------%%%
X = zeros(B,1,chain_num);
for ch = 1:chain_num
    temp = beta_chain{ch};
    X(:,1,ch) = temp;
end
X = X(:,:,chain_select);
R_beta = zeros(1,B-point_start);
for aa = 1:(B-point_start)
    [R,~,~,~,~] = psrf(X(1:(aa+point_start),:,:));
    R_beta(aa)=R;
end

%%%----------------------------  sigma_r  ----------------------------%%%
X = zeros(B,1,chain_num);
for ch = 1:chain_num
    temp = sigma_r_chain{ch};
    X(:,:,ch) = temp;
end
X = X(:,:,chain_select);
R_sigma_r = zeros(1,B-point_start);
for aa = 1:(B-point_start)
    [R,~,~,~,~] = psrf(X(1:(aa+point_start),:,:));
    R_sigma_r(aa)=R;
end

%%%----------------------------  lambda_x0  ----------------------------%%%
R_lambda_x0 = zeros(d,B-point_start,level_num);

for xx = 1:level_num
    X = zeros(B,dmax,chain_num);
    for ch = 1:chain_num
        temp = lambda_x0_chain{ch};
        for d = 1:dmax
            X(:,d,ch) = temp(d,xx,:); % state d
        end
    end
    X = X(:,:,chain_select);
    for aa = 1:(B-point_start)
        for d = 1:dmax
            [R,~,~,~,~] = psrf(X(1:(aa+point_start),d,:));
            R_lambda_x0(d,aa,xx)=R;
        end   
    end
end


%%%----------------------------  omega  ----------------------------%%%
R_omega = zeros(2,B-point_start,level_num);

for xx = 1:level_num
    X = zeros(B,2,chain_num);
    for ch = 1:chain_num
        temp = omega_storage_all{ch};
        for d = 1:2
            X(:,d,ch) = temp(d,xx,:); % state d
        end
    end
    X = X(:,:,chain_select);
    for aa = 1:(B-point_start)
        for d = 1:2
            [R,~,~,~,~] = psrf(X(1:(aa+point_start),d,:));
            R_omega(d,aa,xx)=R;
        end   
    end
end

%%%----------------------------  alpha_x0  ----------------------------%%%
R_alpha_x0 = zeros(B-point_start,level_num);
X = zeros(B,level_num,chain_num);
for ch = 1:chain_num
    temp = alpha_x0_chain{ch};
    X(:,:,ch) = temp';
end
X = X(:,:,chain_select);
for xx = 1:level_num
    for aa = 1:(B-point_start)
        [R,~,~,~,~] = psrf(X(1:(aa+point_start),xx,:));
        R_alpha_x0(aa,xx)=R;
    end
end

%%%----------------------------  alpha_x1  ----------------------------%%%
R_alpha_x1 = zeros(B-point_start,level_num);
X = zeros(B,level_num,chain_num);
for ch = 1:chain_num
    temp = alpha_x1_chain{ch};
    X(:,:,ch) = temp';
end
X = X(:,:,chain_select);
for xx = 1:level_num
    for aa = 1:(B-point_start)
        [R,~,~,~,~] = psrf(X(1:(aa+point_start),xx,:));
        R_alpha_x1(aa,xx)=R;
    end
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%  PSRF figure  %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%=================== Observed process ===================%
ll = 1;
legend_name=cell(1,30);
figure(fig);

%---- mu ----%
for d = 1:dmax
    plot(burnin+gap*((point_start+1):B),R_mu(d,:),'-.','LineWidth',lineSZ,'Color',color_order(ll,:)');hold on;
    legend_name{ll}=['$\mu_',num2str(d),'$'];
    ll = ll + 1;
end

%---- sigma ----%
for d = 1:dmax
    plot(burnin+gap*((point_start+1):B),R_sigma(d,:),'-.','LineWidth',lineSZ,'Color',color_order(ll,:)');hold on;
    legend_name{ll}=['$\sigma_',num2str(d),'$'];
    ll = ll + 1;
end

%---- beta ----%
plot(burnin+gap*((point_start+1):B),R_beta,'-.','LineWidth',lineSZ,'Color',color_order(ll,:)');hold on; 
legend_name{ll}='$\beta$';
ll = ll + 1;


%---- sigma_r ----%
plot(burnin+gap*((point_start+1):B),R_sigma_r,'-.','LineWidth',lineSZ,'Color',color_order(ll,:)');hold on;
legend_name{ll}='$\sigma_r$';
ll = ll + 1;


%---- reference ----%
if FigOne == 2
    line([burnin,simsize],[1,1],'linestyle','-','LineWidth',lineSZ+2,'Color',[0 0 0]');
    set(gca,'FontSize',fontSZ,'fontname','times')
    ylabel('PSRF','FontSize',fontSZ)
    xlabel('Iterarion','FontSize',fontSZ)
    legend_name{ll} = 'reference';
    if lgd == 1
        legend(legend_name(1:ll),'Interpreter','latex','FontSize',fontSZ,'fontname','times') % ,'Location','northwest'
        legend('boxoff')
    end
    if Ymax~=0
        ylim([0 Ymax])
    end
    fig=fig+1; 
end

%=================== Hidden process ===================%
if FigOne == 2
    ll = 1;
    legend_name=cell(1,20);
    figure(fig);
end


%---- lambda_x0 ----%
for xx = 1:level_num
    for d = 1:dmax
        plot(burnin+gap*((point_start+1):B),R_lambda_x0(d,:,xx),'-.','LineWidth',lineSZ,'Color',color_order(ll,:)');hold on;
        legend_name{ll}=['$\lambda_',num2str(xx),'^0(',num2str(d),')$'];
        ll = ll + 1;
    end
end

%---- omega ----%
for xx = 1:level_num
    for d = 1:1
        plot(burnin+gap*((point_start+1):B),R_omega(d,:,xx),'-.','LineWidth',lineSZ,'Color',color_order(ll,:)');hold on;
        legend_name{ll}=['$\omega_',num2str(xx),'^',num2str(d-1),'$'];
        ll = ll + 1;
    end
end

%---- alpha_x0 ----%
for xx = 1:level_num
    plot(burnin+gap*((point_start+1):B),R_alpha_x0(:,xx),'-.','LineWidth',lineSZ,'Color',color_order(ll,:)');hold on;
    legend_name{ll}=['$\alpha_',num2str(xx),'^0','$'];
    ll = ll + 1;
end

%---- alpha_x1 ----%
for xx = 1:level_num
    plot(burnin+gap*((point_start+1):B),R_alpha_x1(:,xx),'-.','LineWidth',lineSZ,'Color',color_order(ll,:)');hold on;
    legend_name{ll}=['$\alpha_',num2str(xx),'^1','$'];
    ll = ll + 1;
end


%---- reference ----%
line([burnin,simsize],[1,1],'linestyle','-','LineWidth',lineSZ+2,'Color',[0 0 0]');
set(gca,'FontSize',fontSZ,'fontname','times')
ylabel('PSRF','FontSize',fontSZ)
xlabel('Iterarion','FontSize',fontSZ)
legend_name{ll} = 'reference';
if lgd == 1
    legend(legend_name(1:ll),'Interpreter','latex','FontSize',fontSZ,'fontname','times') % ,'Location','northwest'
    legend('boxoff')
end
if Ymax~=0
    ylim([0 Ymax])
end
fig=fig+1; 
