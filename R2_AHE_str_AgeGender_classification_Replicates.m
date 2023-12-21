%%%%%% Case Study: MIMIC-III Matched data %%%%%%
%%%%%% Acute Hypotension Episode prediction
%%%%%% Covariate: Age + Gender
clear;clc;close all;
% dbstop if error
%% Set seed 
NRep = 48; % Number of replications
seed = (1 : NRep) + 60; % Set random seeds

% dbstop if error;

%% Parallel implementation
% parfor ss = 1:NRep
for ss = 1:1  
    FinalModelRep(seed(ss));
end


%%
function [] = FinalModelRep(seed)

rng(seed);
RandStream.getGlobalStream;

%% load data and parameters
% 2, 7, 8
method='Mean'; % 'Raw'; 'Mean'; 'MeanVar'
seed_opt=11;

filename0=['Result_AgeGender/CV/',method,'_Final_data_Seed_',num2str(seed_opt),'.mat'];
load(filename0,'str_id0','Str_num','model_str_best0',...
'idx_A_Gap0','Train_Num_N','T','dmax','qmax','N1','level_N',...
'KK_level1','KK_level2','Y_A_K_Gap0','Y_total_A_Gap0','T_A_K',...
'Y_NA_K_Gap0','Y_total_NA_Gap0','T_NA_K',...
'alpha_x','alpha00','Parameters_A_Gap0',...
'likelihood_type','Parameters_prior_A_Gap0',...
'Parameters_NA_Gap0',...
'Parameters_prior_NA_Gap0','pM','simsize',...
'burnin','gap','pigamma','alpha_x0','alpha_x1','alpha00','alpha_a0',...
'alpha_b0','alpha_a1','alpha_b1','omega','omega_a','omega_b',...
'idx_NA_Gap0','KK_level_t','Y_raw_t_Gap0',...
'KK_A_t','KK_NA_t')
  
%% File save (classification + parameter estimation)

% classification
results_final=nan(3,6+1);
% AUC & Accuracy & Sensivity & Specificity & Precision & F1-score & Time; 
% Best_MHOHMM, HOHMM, MHMM

% parameter estimation
% Gap X class X models
Mact_figure=cell(2,2,3); % order
weight_figure=cell(2,2,3); % weight
lambda_x0_figure=cell(2,2,3); % state preference
mu_figure=cell(2,2,3);  % mean
sigma_figure=cell(2,2,3); % variance
beta_figure=cell(2,2,3); % fixed effect
Sigma_r_figure=cell(2,2,3); % random effect

%%  Gap 0 (Final model -- MHOHMM)
tic
%%%%%%%%%%%%%%%
%%%   AHE   %%%
%%%%%%%%%%%%%%%
beta=0.5; % initial fixed effect
C_total=idx_A_Gap0;
CC_K=cell(1,Train_Num_N); % initial state sequences 
for aa=1:Train_Num_N
    CC_K{aa}=C_total((T*(aa-1)+1):(T*aa));
end

%%% Sampler for Stage 1
fprintf('\n--- MHOHMM (Model AHE; Gap0) --- \n\n');
fprintf('\n--- First Stage --- \n\n');
[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Train_Num_N,level_N,KK_level1,CC_K,Y_A_K_Gap0,...
    Y_total_A_Gap0,T_A_K,alpha_x,alpha00,Parameters_A_Gap0,...
    beta,likelihood_type,Parameters_prior_A_Gap0,pM,model_str_best0);

%%% Sampler for Stage 2
fprintf('\n --- Second Stage --- \n\n');
pause(1);
[lambda_x_Mix_N,lambda_x0_Mix_N,alpha_x0_Mix_N,alpha_x1_Mix_N,omega_Mix_N,...
    pi_Mix_N,mu_Mix_N,sigma_Mix_N,beta_Mix_N,Sigma_r_Mix_N,L00_Mix_N,...
    State_select_Mix_N,p00_Mix_N,K00_Mix_N,ind00_Mix_N,likeli_Mix_N]=...
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Train_Num_N,...
    level_N,KK_level1,C_total,CC_K,Xnew,Cnew,Y_A_K_Gap0,Y_total_A_Gap0,...
    T_A_K,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,....
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_A_Gap0,model_str_best0);

% save parameters
Mact_figure{1,1,1}=Mact_L;
weight_figure{1,1,1}=omega_Mix_N;
lambda_x0_figure{1,1,1}=lambda_x0_Mix_N;
mu_figure{1,1,1}=mu_Mix_N;
sigma_figure{1,1,1}=sigma_Mix_N;
beta_figure{1,1,1}=beta_Mix_N;
Sigma_r_figure{1,1,1}=Sigma_r_Mix_N;


%%%%%%%%%%%%%%%%%%%
%%%   Non-AHE   %%%
%%%%%%%%%%%%%%%%%%%

beta=0.5; % initial fixed effect
C_total=idx_NA_Gap0;
CC_K=cell(1,Train_Num_N); % initial state sequences 
for aa=1:Train_Num_N
    CC_K{aa}=C_total((T*(aa-1)+1):(T*aa));
end

%%% Sampler for Stage 1
fprintf('\n--- MHOHMM (Model Non-AHE; Gap0) --- \n\n');
fprintf('\n--- First Stage --- \n\n');
[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Train_Num_N,level_N,KK_level2,CC_K,Y_NA_K_Gap0,...
    Y_total_NA_Gap0,T_NA_K,alpha_x,alpha00,Parameters_NA_Gap0,...
    beta,likelihood_type,Parameters_prior_NA_Gap0,pM,model_str_best0);

%%% Sampler for Stage 2
fprintf('\n --- Second Stage --- \n\n');
pause(1);
[lambda_x_Mix_NA,lambda_x0_Mix_NA,alpha_x0_Mix_NA,alpha_x1_Mix_NA,omega_Mix_NA,pi_Mix_NA,...
    mu_Mix_NA,sigma_Mix_NA,beta_Mix_NA,Sigma_r_Mix_NA,L00_Mix_NA,...
    State_select_Mix_NA,p00_Mix_NA,K00_Mix_NA,ind00_Mix_NA,likeli_Mix_NA]=...
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Train_Num_N,...
    level_N,KK_level2,C_total,CC_K,Xnew,Cnew,Y_NA_K_Gap0,Y_total_NA_Gap0,...
    T_NA_K,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,...
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_NA_Gap0,model_str_best0);

% save parameters
Mact_figure{1,2,1}=Mact_L;
weight_figure{1,2,1}=omega_Mix_NA;
lambda_x0_figure{1,2,1}=lambda_x0_Mix_NA;
mu_figure{1,2,1}=mu_Mix_NA;
sigma_figure{1,2,1}=sigma_Mix_NA;
beta_figure{1,2,1}=beta_Mix_NA;
Sigma_r_figure{1,2,1}=Sigma_r_Mix_NA;

%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Classification  %%%
%%%%%%%%%%%%%%%%%%%%%%%%
sampsize=1000;
num_para=3;
KK_t=length(KK_level_t);
Y_Test=Y_raw_t_Gap0;
classTrue=[ones(1,KK_A_t),repmat(2,1,KK_NA_t)];
class_test_MHOHMM=ones(1,KK_t); 
scoreN_MHOHMM=ones(1,KK_t); 
scoreNA_MHOHMM=ones(1,KK_t);
disp('--- MHOHMM Classification ---');  
for kk=1:KK_t
    group=KK_level_t(kk);
    data=Y_Test{kk};

    % Model N %
    [LogLikeli_N,~,~]=decode_MHOHMM_str_aoas(data,group,lambda_x_Mix_N,...
        lambda_x0_Mix_N,alpha_x0_Mix_N,omega_Mix_N,pi_Mix_N,mu_Mix_N,...
        sigma_Mix_N,beta_Mix_N,Sigma_r_Mix_N,likelihood_type,qmax,dmax,...
        sampsize,p00_Mix_N,K00_Mix_N,ind00_Mix_N,num_para,model_str_best0);
    % Model NA %
    [LogLikeli_NA,~,~]=decode_MHOHMM_str_aoas(data,group,lambda_x_Mix_NA,...
        lambda_x0_Mix_NA,alpha_x0_Mix_NA,omega_Mix_NA,pi_Mix_NA,mu_Mix_NA,...
        sigma_Mix_NA,beta_Mix_NA,Sigma_r_Mix_NA,likelihood_type,qmax,dmax,...
        sampsize,p00_Mix_NA,K00_Mix_NA,ind00_Mix_NA,num_para,model_str_best0);

    if mean(LogLikeli_N)<mean(LogLikeli_NA)
        class_test_MHOHMM(kk)=2; % NA class
    end
    scoreN_MHOHMM(kk)=mean(LogLikeli_N);
    scoreNA_MHOHMM(kk)=mean(LogLikeli_NA);
end
toc
[~,~,~,auc_M]=perfcurve(classTrue,(scoreN_MHOHMM-scoreNA_MHOHMM),1); % MHOHMM
results_final(1,1)=auc_M;
% 1: AHE; 2: Non-AHE
TP = sum((class_test_MHOHMM==1).*(classTrue==1)); 
FN = sum((class_test_MHOHMM==2).*(classTrue==1));
TN = sum((class_test_MHOHMM==2).*(classTrue==2));
FP = sum((class_test_MHOHMM==1).*(classTrue==2));
results_final(1,2) = (TP+TN)/(TP+FN+TN+FP); % Acc
results_final(1,3) = TP/(TP+FN); % Sensivity
results_final(1,4) = TN/(TN+FP); % Specificity
results_final(1,5) = TP/(TP+FP); % Precision
results_final(1,6) = (2*TP)/(2*TP+FP+FN); % F1-score
results_final(1,7) = toc/60/60; % hours

%%  Gap 0
tic
%%%%%%%%%%%%%%%%%%%%%%
%%% HOHMM learning %%%
%%%%%%%%%%%%%%%%%%%%%%

model_str_HOHMM=[0,0,0,0,1];

%%%%%%%%%%%%%%%
%%%   AHE   %%%
%%%%%%%%%%%%%%%
beta=0.5; % initial fixed effect
C_total=idx_A_Gap0;
CC_K=cell(1,Train_Num_N); % initial state sequences 
for aa=1:Train_Num_N
    CC_K{aa}=C_total((T*(aa-1)+1):(T*aa));
end

%%% Sampler for Stage 1
fprintf('\n--- HOHMM (Model AHE; Gap0) --- \n\n');
fprintf('\n--- First Stage --- \n\n');
[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Train_Num_N,level_N,KK_level1,CC_K,Y_A_K_Gap0,...
    Y_total_A_Gap0,T_A_K,alpha_x,alpha00,Parameters_A_Gap0,...
    beta,likelihood_type,Parameters_prior_A_Gap0,pM,model_str_HOHMM);

%%% Sampler for Stage 2
fprintf('\n --- Second Stage --- \n\n');
pause(1);
[lambda_x_Mix_N,lambda_x0_Mix_N,alpha_x0_Mix_N,alpha_x1_Mix_N,omega_Mix_N,...
    pi_Mix_N,mu_Mix_N,sigma_Mix_N,beta_Mix_N,Sigma_r_Mix_N,L00_Mix_N,...
    State_select_Mix_N,p00_Mix_N,K00_Mix_N,ind00_Mix_N,likeli_Mix_N]=...
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Train_Num_N,...
    level_N,KK_level1,C_total,CC_K,Xnew,Cnew,Y_A_K_Gap0,Y_total_A_Gap0,...
    T_A_K,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,....
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_A_Gap0,model_str_HOHMM);

% save parameters
Mact_figure{1,1,2}=Mact_L;
weight_figure{1,1,2}=omega_Mix_N;
lambda_x0_figure{1,1,2}=lambda_x0_Mix_N;
mu_figure{1,1,2}=mu_Mix_N;
sigma_figure{1,1,2}=sigma_Mix_N;
beta_figure{1,1,2}=beta_Mix_N;
Sigma_r_figure{1,1,2}=Sigma_r_Mix_N;

%%%%%%%%%%%%%%%%%%%
%%%   Non-AHE   %%%
%%%%%%%%%%%%%%%%%%%

beta=0.5; % initial fixed effect
C_total=idx_NA_Gap0;
CC_K=cell(1,Train_Num_N); % initial state sequences 
for aa=1:Train_Num_N
    CC_K{aa}=C_total((T*(aa-1)+1):(T*aa));
end

%%% Sampler for Stage 1
fprintf('\n--- HOHMM (Model Non-AHE; Gap0) --- \n\n');
fprintf('\n--- First Stage --- \n\n');
[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Train_Num_N,level_N,KK_level2,CC_K,Y_NA_K_Gap0,...
    Y_total_NA_Gap0,T_NA_K,alpha_x,alpha00,Parameters_NA_Gap0,...
    beta,likelihood_type,Parameters_prior_NA_Gap0,pM,model_str_HOHMM);

%%% Sampler for Stage 2
fprintf('\n --- Second Stage --- \n\n');
pause(1);
[lambda_x_Mix_NA,lambda_x0_Mix_NA,alpha_x0_Mix_NA,alpha_x1_Mix_NA,omega_Mix_NA,pi_Mix_NA,...
    mu_Mix_NA,sigma_Mix_NA,beta_Mix_NA,Sigma_r_Mix_NA,L00_Mix_NA,...
    State_select_Mix_NA,p00_Mix_NA,K00_Mix_NA,ind00_Mix_NA,likeli_Mix_NA]=...
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Train_Num_N,...
    level_N,KK_level2,C_total,CC_K,Xnew,Cnew,Y_NA_K_Gap0,Y_total_NA_Gap0,...
    T_NA_K,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,...
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_NA_Gap0,model_str_HOHMM);

% save parameters
Mact_figure{1,2,2}=Mact_L;
weight_figure{1,2,2}=omega_Mix_NA;
lambda_x0_figure{1,2,2}=lambda_x0_Mix_NA;
mu_figure{1,2,2}=mu_Mix_NA;
sigma_figure{1,2,2}=sigma_Mix_NA;
beta_figure{1,2,2}=beta_Mix_NA;
Sigma_r_figure{1,2,2}=Sigma_r_Mix_NA;

%%% Classification HOHMM %%%
class_test_HOHMM=ones(1,KK_t); 
scoreN_HOHMM=ones(1,KK_t); 
scoreNA_HOHMM=ones(1,KK_t);
disp('--- HOHMM Classification ---');  
for kk=1:KK_t
    group=KK_level_t(kk);
    data=Y_Test{kk};

    % Model N %
    [LogLikeli_N,~,~]=decode_MHOHMM_str_aoas(data,group,lambda_x_Mix_N,...
        lambda_x0_Mix_N,alpha_x0_Mix_N,omega_Mix_N,pi_Mix_N,mu_Mix_N,...
        sigma_Mix_N,beta_Mix_N,Sigma_r_Mix_N,likelihood_type,qmax,dmax,...
        sampsize,p00_Mix_N,K00_Mix_N,ind00_Mix_N,num_para,model_str_HOHMM);
    % Model NA %
    [LogLikeli_NA,~,~]=decode_MHOHMM_str_aoas(data,group,lambda_x_Mix_NA,...
        lambda_x0_Mix_NA,alpha_x0_Mix_NA,omega_Mix_NA,pi_Mix_NA,mu_Mix_NA,...
        sigma_Mix_NA,beta_Mix_NA,Sigma_r_Mix_NA,likelihood_type,qmax,dmax,...
        sampsize,p00_Mix_NA,K00_Mix_NA,ind00_Mix_NA,num_para,model_str_HOHMM);

    if mean(LogLikeli_N)<mean(LogLikeli_NA)
        class_test_HOHMM(kk)=2; % NA class
    end
    scoreN_HOHMM(kk)=mean(LogLikeli_N);
    scoreNA_HOHMM(kk)=mean(LogLikeli_NA);
end
toc
[~,~,~,auc_H]=perfcurve(classTrue,(scoreN_HOHMM-scoreNA_HOHMM),1); % MHOHMM
results_final(2,1)=auc_H;
% 1: AHE; 2: Non-AHE
TP = sum((class_test_HOHMM==1).*(classTrue==1)); 
FN = sum((class_test_HOHMM==2).*(classTrue==1));
TN = sum((class_test_HOHMM==2).*(classTrue==2));
FP = sum((class_test_HOHMM==1).*(classTrue==2));
results_final(2,2) = (TP+TN)/(TP+FN+TN+FP); % Acc
results_final(2,3) = TP/(TP+FN); % Sensivity
results_final(2,4) = TN/(TN+FP); % Specificity
results_final(2,5) = TP/(TP+FP); % Precision
results_final(2,6) = (2*TP)/(2*TP+FP+FN); % F1-score
results_final(2,7) = toc/60/60; % hours

%% Gap 0
tic
%%%%%%%%%%%%%%%%%%%%%
%%% MHMM learning %%%
%%%%%%%%%%%%%%%%%%%%%%
model_str_MHMM=model_str_best0;
model_str_MHMM(end)=0;

%%%%%%%%%%%%%%%
%%%   AHE   %%%
%%%%%%%%%%%%%%%
beta=0.5; % initial fixed effect
C_total=idx_A_Gap0;
CC_K=cell(1,Train_Num_N); % initial state sequences 
for aa=1:Train_Num_N
    CC_K{aa}=C_total((T*(aa-1)+1):(T*aa));
end

%%% Sampler for Stage 1
fprintf('\n--- MHMM (Model Non-AHE; Gap0) --- \n\n');
fprintf('\n--- First Stage --- \n\n');

[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Train_Num_N,level_N,KK_level1,CC_K,Y_A_K_Gap0,...
    Y_total_A_Gap0,T_A_K,alpha_x,alpha00,Parameters_A_Gap0,...
    beta,likelihood_type,Parameters_prior_A_Gap0,pM,model_str_MHMM);

%%% Sampler for Stage 2
fprintf('\n --- Second Stage --- \n\n');
pause(1);
[lambda_x_Mix_N,lambda_x0_Mix_N,alpha_x0_Mix_N,alpha_x1_Mix_N,omega_Mix_N,...
    pi_Mix_N,mu_Mix_N,sigma_Mix_N,beta_Mix_N,Sigma_r_Mix_N,L00_Mix_N,...
    State_select_Mix_N,p00_Mix_N,K00_Mix_N,ind00_Mix_N,likeli_Mix_N]=...
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Train_Num_N,...
    level_N,KK_level1,C_total,CC_K,Xnew,Cnew,Y_A_K_Gap0,Y_total_A_Gap0,...
    T_A_K,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,....
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_A_Gap0,model_str_MHMM);

% save parameters
Mact_figure{1,1,3}=Mact_L;
weight_figure{1,1,3}=omega_Mix_N;
lambda_x0_figure{1,1,3}=lambda_x0_Mix_N;
mu_figure{1,1,3}=mu_Mix_N;
sigma_figure{1,1,3}=sigma_Mix_N;
beta_figure{1,1,3}=beta_Mix_N;
Sigma_r_figure{1,1,3}=Sigma_r_Mix_N;

%%%%%%%%%%%%%%%%%%%
%%%   Non-AHE   %%%
%%%%%%%%%%%%%%%%%%%

beta=0.5; % initial fixed effect
C_total=idx_NA_Gap0;
CC_K=cell(1,Train_Num_N); % initial state sequences 
for aa=1:Train_Num_N
    CC_K{aa}=C_total((T*(aa-1)+1):(T*aa));
end

%%% Sampler for Stage 1
fprintf('\n--- MHMM (Model Non-AHE; Gap0) --- \n\n');
fprintf('\n--- First Stage --- \n\n');
[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Train_Num_N,level_N,KK_level2,CC_K,Y_NA_K_Gap0,...
    Y_total_NA_Gap0,T_NA_K,alpha_x,alpha00,Parameters_NA_Gap0,...
    beta,likelihood_type,Parameters_prior_NA_Gap0,pM,model_str_MHMM);

%%% Sampler for Stage 2
fprintf('\n --- Second Stage --- \n\n');
pause(1);
[lambda_x_Mix_NA,lambda_x0_Mix_NA,alpha_x0_Mix_NA,alpha_x1_Mix_NA,omega_Mix_NA,pi_Mix_NA,...
    mu_Mix_NA,sigma_Mix_NA,beta_Mix_NA,Sigma_r_Mix_NA,L00_Mix_NA,...
    State_select_Mix_NA,p00_Mix_NA,K00_Mix_NA,ind00_Mix_NA,likeli_Mix_NA]=...
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Train_Num_N,...
    level_N,KK_level2,C_total,CC_K,Xnew,Cnew,Y_NA_K_Gap0,Y_total_NA_Gap0,...
    T_NA_K,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,...
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_NA_Gap0,model_str_MHMM);

% save parameters
Mact_figure{1,2,3}=Mact_L;
weight_figure{1,2,3}=omega_Mix_NA;
lambda_x0_figure{1,2,3}=lambda_x0_Mix_NA;
mu_figure{1,2,3}=mu_Mix_NA;
sigma_figure{1,2,3}=sigma_Mix_NA;
beta_figure{1,2,3}=beta_Mix_NA;
Sigma_r_figure{1,2,3}=Sigma_r_Mix_NA;

%%% Classification MHMM %%%
class_test_MHMM=ones(1,KK_t); 
scoreN_MHMM=ones(1,KK_t); 
scoreNA_MHMM=ones(1,KK_t);
disp('--- MHMM Classification ---');
for kk=1:KK_t
    group=KK_level_t(kk);
    data=Y_Test{kk};

    % Model N %
    [LogLikeli_N,~,~]=decode_MHOHMM_str_aoas(data,group,lambda_x_Mix_N,...
        lambda_x0_Mix_N,alpha_x0_Mix_N,omega_Mix_N,pi_Mix_N,mu_Mix_N,...
        sigma_Mix_N,beta_Mix_N,Sigma_r_Mix_N,likelihood_type,qmax,dmax,...
        sampsize,p00_Mix_N,K00_Mix_N,ind00_Mix_N,num_para,model_str_MHMM);
    % Model NA %
    [LogLikeli_NA,~,~]=decode_MHOHMM_str_aoas(data,group,lambda_x_Mix_NA,...
        lambda_x0_Mix_NA,alpha_x0_Mix_NA,omega_Mix_NA,pi_Mix_NA,mu_Mix_NA,...
        sigma_Mix_NA,beta_Mix_NA,Sigma_r_Mix_NA,likelihood_type,qmax,dmax,...
        sampsize,p00_Mix_NA,K00_Mix_NA,ind00_Mix_NA,num_para,model_str_MHMM);

    if mean(LogLikeli_N)<mean(LogLikeli_NA)
        class_test_MHMM(kk)=2; % NA class
    end
    scoreN_MHMM(kk)=mean(LogLikeli_N);
    scoreNA_MHMM(kk)=mean(LogLikeli_NA);
end
toc
[~,~,~,auc_m]=perfcurve(classTrue,(scoreN_MHMM-scoreNA_MHMM),1); % MHOHMM
results_final(3,1)=auc_m;    
TP = sum((class_test_MHMM==1).*(classTrue==1)); 
FN = sum((class_test_MHMM==2).*(classTrue==1));
TN = sum((class_test_MHMM==2).*(classTrue==2));
FP = sum((class_test_MHMM==1).*(classTrue==2));
results_final(3,2) = (TP+TN)/(TP+FN+TN+FP); % Acc
results_final(3,3) = TP/(TP+FN); % Sensivity
results_final(3,4) = TN/(TN+FP); % Specificity
results_final(3,5) = TP/(TP+FP); % Precision
results_final(3,6) = (2*TP)/(2*TP+FP+FN); % F1-score
results_final(3,7) = toc/60/60; % hours

%%
%%%%%%%%%%%%%%%%%%%
%%%%% Results %%%%%
%%%%%%%%%%%%%%%%%%%
% Classification
Model_Name={'Hf(Gap0)';'Hfr(Gap0)';'Hf.Of(Gap0)';'Hfr.Of(Gap0)';...
    'Hf.Or(Gap0)';'Hfr.Or(Gap0)';'Hf.Ofr(Gap0)';'Hfr.Ofr(Gap0)'};
%%% Testing set %%%
Model={Model_Name{str_id0};'HOHMM(Gap0)';'MHMM(Gap0)'};
temp=table(Model);
results=array2table(results_final,'VariableNames',{'AUC','Accuracy',...
    'Sensivity','Specificity','Precision','F1-score','Time(hrs)'});
filename2=['Result_AgeGender/Replicates/Seed_',num2str(seed_opt),...
    '/Final_classification_test_Replicate_',num2str(seed),'.xlsx'];
writetable([temp,results],filename2);

% Parameter estimation
%%% save parameters for future use
filename3=['Result_AgeGender/Replicates/Seed_',num2str(seed_opt),...
    '/ParaEst_ForFigure_Seed_',num2str(seed),'.mat'];
save(filename3,'Model',...
    'Mact_figure','weight_figure','lambda_x0_figure',...
    'mu_figure','sigma_figure','beta_figure','Sigma_r_figure')


end

