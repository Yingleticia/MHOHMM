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

% NRep = 1; % Number of replications
% seed = (1 : NRep) + 1; % Set random seeds
%% Set seed 
NRep = 48; % Number of replications
seed = (1 : NRep) + 0; % Set random seeds

level_N = 2; % number of groups: 2,4,6,8,10 (sensitivity analysis)
seed_opt = 2;
% 2 - 2 (2, 12) 
% 3 - 11
% 4 - 7 
% 5 - 24 (2, 24)
% 6 - 18 (12, 18)

SenAna_q_d = 1; % 1: sensitivity analysis for qmax and dmax

Allmodel = 0; % 1: MHOHMM + HOHMM + MHMM; 0: MHOHMM only
qmax = 5; % 5,7,9 (sensitivity analysis); default: 5
dmax = 7; % 3,5,7 (sensitivity analysis); default: 3

if SenAna_q_d == 0
    folder = ['Result_AOAS/Classification/Revision/Groups_',num2str(level_N),'/'];
    qmax = 5;
    dmax = 3;
    Allmodel = 1;
else
    folder = ['Result_AOAS/Classification/Revision/Groups_',num2str(level_N),'/All_q_d/q',num2str(qmax),'_d',num2str(dmax),'/'];
    level_N = 2;
    seed_opt = 2;
end

% dbstop if error;

%% Parallel implementation
parfor ss = 1:NRep
% for irep = 1:NRep  
    FinalModelRep(seed(ss),dmax,qmax,folder,seed_opt,SenAna_q_d,Allmodel);
end


%%
function [] = FinalModelRep(seed,dmax,qmax,folder,seed_opt,SenAna_q_d,Allmodel)

rng(seed);
RandStream.getGlobalStream;

results_final=nan(3,7);
% AUC & Accuracy & Sensivity & Specificity & Precision & F1-score & computation time;  
% Best_MHOHMM, HOHMM, MHMM

%% load data and parameters
% 18, 24
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
 
%% Final Model (Hfr.Ofr)
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

% [Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
%     lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1(dmax,qmax,N1,...
%     Train_Num_N,level_N,level_Train_N,CC_K,Y_Train_N,...
%     Y_Train_total_N,T_N,alpha_x,alpha00,Parameters_N,mu_r_K,beta,...
%     likelihood_type,Parameters_prior_N,pM,model_str);

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
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Train_Num_N,...
    level_N,level_Train_N,C_total,CC_K,Xnew,Cnew,Y_Train_N,Y_Train_total_N,...
    T_N,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,....
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_N,model_str_best);

%%
%%%%%%%%%%%%%%%%
%%% Model NA %%%
%%%%%%%%%%%%%%%%
%%%%%%%%%% Stage 1 %%%%%%%%%%%%
%%% Assign Priors 
% emission distribution

beta=0.5; % initial fixed effect
C_total=idx_NA;
CC_K=cell(1,Train_Num_NA); % initial state sequences 
for aa=1:Train_Num_NA
    CC_K{aa}=C_total((T*(aa-1)+1):(T*aa));
end

%%% Sampler for Stage 1
fprintf('\n--- MHOHMM (Model NA) --- \n\n');
fprintf('\n--- First Stage --- \n\n');
% 
% [Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
%     lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1(dmax,qmax,N1,...
%     Train_Num_NA,level_N,level_Train_NA,CC_K,Y_Train_NA,...
%     Y_Train_total_NA,T_NA,alpha_x,alpha00,Parameters_NA,mu_r_K,beta,...
%     likelihood_type,Parameters_prior_NA,pM,model_str);

[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Train_Num_NA,level_N,level_Train_NA,CC_K,Y_Train_NA,...
    Y_Train_total_NA,T_NA,alpha_x,alpha00,Parameters_NA,beta,...
    likelihood_type,Parameters_prior_NA,pM,model_str_best);


%%%%%%%%%% Stage 2 %%%%%%%%%%%%
%%% Assign Priors & initial values

%%% Sampler for Stage 2
fprintf('\n --- Second Stage --- \n\n');
pause(1);
[lambda_x_Mix_NA,lambda_x0_Mix_NA,alpha_x0_Mix_NA,alpha_x1_Mix_NA,omega_Mix_NA,pi_Mix_NA,...
    mu_Mix_NA,sigma_Mix_NA,beta_Mix_NA,Sigma_r_Mix_NA,L00_Mix_NA,...
    State_select_Mix_NA,p00_Mix_NA,K00_Mix_NA,ind00_Mix_NA,likeli_Mix_NA]=...
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Train_Num_NA,...
    level_N,level_Train_NA,C_total,CC_K,Xnew,Cnew,Y_Train_NA,Y_Train_total_NA,...
    T_NA,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,...
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_NA,model_str_best);




%% classification

sampsize=1000;
num_para=3;
KK_level_t=[level_Test_N,level_Test_NA]; % level/group for testing units
KK_t=length(KK_level_t); % num of testing units
Y_Test=[Y_Test_N,Y_Test_NA]; % data for testing
classTrue=[ones(1,Test_Num_N),repmat(2,1,Test_Num_NA)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Classification MHOHMM %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        sampsize,p00_Mix_N,K00_Mix_N,ind00_Mix_N,num_para,model_str_best);
    % Model NA %
    [LogLikeli_NA,~,~]=decode_MHOHMM_str_aoas(data,group,lambda_x_Mix_NA,...
        lambda_x0_Mix_NA,alpha_x0_Mix_NA,omega_Mix_NA,pi_Mix_NA,mu_Mix_NA,...
        sigma_Mix_NA,beta_Mix_NA,Sigma_r_Mix_NA,likelihood_type,qmax,dmax,...
        sampsize,p00_Mix_NA,K00_Mix_NA,ind00_Mix_NA,num_para,model_str_best);

    if mean(LogLikeli_N)<mean(LogLikeli_NA)
        class_test_MHOHMM(kk)=2; % NA class
    end
    scoreN_MHOHMM(kk)=mean(LogLikeli_N);
    scoreNA_MHOHMM(kk)=mean(LogLikeli_NA);
end
[~,~,~,auc_M]=perfcurve(classTrue,(scoreN_MHOHMM-scoreNA_MHOHMM),1); % MHOHMM
results_final(1,1)=auc_M;
% 1: N; 2: NA
TP = sum((class_test_MHOHMM==2).*(classTrue==2)); 
FN = sum((class_test_MHOHMM==1).*(classTrue==2));
TN = sum((class_test_MHOHMM==1).*(classTrue==1));
FP = sum((class_test_MHOHMM==2).*(classTrue==1));
results_final(1,2) = (TP+TN)/(TP+FN+TN+FP); % Acc
results_final(1,3) = TP/(TP+FN); % Sensivity
results_final(1,4) = TN/(TN+FP); % Specificity
results_final(1,5) = TP/(TP+FP); % Precision
results_final(1,6) = (2*TP)/(2*TP+FP+FN); % F1-score

comp_time = toc;
results_final(1,7) = comp_time/60/60; % hours



%%
%%%%%%%%%%%%%%%%%%%%%%
%%% HOHMM learning %%%
%%%%%%%%%%%%%%%%%%%%%%
if Allmodel == 1
    
model_str_HOHMM=[0,0,0,0,1];

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
fprintf('\n--- HOHMM (Model N) --- \n\n');
fprintf('\n--- First Stage --- \n\n');


[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Train_Num_N,level_N,level_Train_N,CC_K,Y_Train_N,...
    Y_Train_total_N,T_N,alpha_x,alpha00,...
    Parameters_N,beta,likelihood_type,Parameters_prior_N,pM,model_str_HOHMM);

%%%%%%%%%% Stage 2 %%%%%%%%%%%%
%%% Sampler for Stage 2
fprintf('\n --- Second Stage --- \n\n');
pause(1);

[lambda_x_Mix_N,lambda_x0_Mix_N,alpha_x0_Mix_N,alpha_x1_Mix_N,omega_Mix_N,...
    pi_Mix_N,mu_Mix_N,sigma_Mix_N,beta_Mix_N,Sigma_r_Mix_N,L00_Mix_N,...
    State_select_Mix_N,p00_Mix_N,K00_Mix_N,ind00_Mix_N,likeli_Mix_N]=...
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Train_Num_N,...
    level_N,level_Train_N,C_total,CC_K,Xnew,Cnew,Y_Train_N,Y_Train_total_N,...
    T_N,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,....
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_N,model_str_HOHMM);


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
fprintf('\n--- HOHMM (Model NA) --- \n\n');
fprintf('\n--- First Stage --- \n\n');

[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Train_Num_NA,level_N,level_Train_NA,CC_K,Y_Train_NA,...
    Y_Train_total_NA,T_NA,alpha_x,alpha00,Parameters_NA,beta,...
    likelihood_type,Parameters_prior_NA,pM,model_str_HOHMM);


%%%%%%%%%% Stage 2 %%%%%%%%%%%%
%%% Sampler for Stage 2
fprintf('\n --- Second Stage --- \n\n');
pause(1);

[lambda_x_Mix_NA,lambda_x0_Mix_NA,alpha_x0_Mix_NA,alpha_x1_Mix_NA,omega_Mix_NA,pi_Mix_NA,...
    mu_Mix_NA,sigma_Mix_NA,beta_Mix_NA,Sigma_r_Mix_NA,L00_Mix_NA,...
    State_select_Mix_NA,p00_Mix_NA,K00_Mix_NA,ind00_Mix_NA,likeli_Mix_NA]=...
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Train_Num_NA,...
    level_N,level_Train_NA,C_total,CC_K,Xnew,Cnew,Y_Train_NA,Y_Train_total_NA,...
    T_NA,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,...
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_NA,model_str_HOHMM);
 


%% Classification (HOHMM)

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

[~,~,~,auc_H]=perfcurve(classTrue,(scoreN_HOHMM-scoreNA_HOHMM),1); % MHOHMM
results_final(2,1)=auc_H;
% 1: N; 2: NA
TP = sum((class_test_HOHMM==2).*(classTrue==2)); 
FN = sum((class_test_HOHMM==1).*(classTrue==2));
TN = sum((class_test_HOHMM==1).*(classTrue==1));
FP = sum((class_test_HOHMM==2).*(classTrue==1));
results_final(2,2) = (TP+TN)/(TP+FN+TN+FP); % Acc
results_final(2,3) = TP/(TP+FN); % Sensivity
results_final(2,4) = TN/(TN+FP); % Specificity
results_final(2,5) = TP/(TP+FP); % Precision
results_final(2,6) = (2*TP)/(2*TP+FP+FN); % F1-score

comp_time = toc;
results_final(2,7) = comp_time/60/60; % hours


%%
%%%%%%%%%%%%%%%%%%%%%
%%% MHMM learning %%%
%%%%%%%%%%%%%%%%%%%%%%
model_str_MHMM=model_str_best;
model_str_MHMM(5)=0;

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
fprintf('\n--- MHMM (Model N) --- \n\n');
fprintf('\n--- First Stage --- \n\n');
alpha0=1;

[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Train_Num_N,level_N,level_Train_N,CC_K,Y_Train_N,...
    Y_Train_total_N,T_N,alpha_x,alpha00,...
    Parameters_N,beta,likelihood_type,Parameters_prior_N,pM,model_str_MHMM);

%%%%%%%%%% Stage 2 %%%%%%%%%%%%
%%% Sampler for Stage 2
fprintf('\n --- Second Stage --- \n\n');
pause(1);

[lambda_x_Mix_N,lambda_x0_Mix_N,alpha_x0_Mix_N,alpha_x1_Mix_N,omega_Mix_N,...
    pi_Mix_N,mu_Mix_N,sigma_Mix_N,beta_Mix_N,Sigma_r_Mix_N,L00_Mix_N,...
    State_select_Mix_N,p00_Mix_N,K00_Mix_N,ind00_Mix_N,likeli_Mix_N]=...
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Train_Num_N,...
    level_N,level_Train_N,C_total,CC_K,Xnew,Cnew,Y_Train_N,Y_Train_total_N,...
    T_N,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,....
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_N,model_str_MHMM);

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
fprintf('\n--- MHMM (Model NA) --- \n\n');
fprintf('\n--- First Stage --- \n\n');

[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Train_Num_NA,level_N,level_Train_NA,CC_K,Y_Train_NA,...
    Y_Train_total_NA,T_NA,alpha_x,alpha00,Parameters_NA,beta,...
    likelihood_type,Parameters_prior_NA,pM,model_str_MHMM);

%%%%%%%%%% Stage 2 %%%%%%%%%%%%
%%% Sampler for Stage 2
fprintf('\n --- Second Stage --- \n\n');
pause(1);

[lambda_x_Mix_NA,lambda_x0_Mix_NA,alpha_x0_Mix_NA,alpha_x1_Mix_NA,omega_Mix_NA,pi_Mix_NA,...
    mu_Mix_NA,sigma_Mix_NA,beta_Mix_NA,Sigma_r_Mix_NA,L00_Mix_NA,...
    State_select_Mix_NA,p00_Mix_NA,K00_Mix_NA,ind00_Mix_NA,likeli_Mix_NA]=...
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Train_Num_NA,...
    level_N,level_Train_NA,C_total,CC_K,Xnew,Cnew,Y_Train_NA,Y_Train_total_NA,...
    T_NA,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,...
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_NA,model_str_MHMM);


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

[~,~,~,auc_H]=perfcurve(classTrue,(scoreN_MHMM-scoreNA_MHMM),1); % MHOHMM
results_final(3,1)=auc_H;
% 1: N; 2: NA
TP = sum((class_test_MHMM==2).*(classTrue==2)); 
FN = sum((class_test_MHMM==1).*(classTrue==2));
TN = sum((class_test_MHMM==1).*(classTrue==1));
FP = sum((class_test_MHMM==2).*(classTrue==1));
results_final(3,2) = (TP+TN)/(TP+FN+TN+FP); % Acc
results_final(3,3) = TP/(TP+FN); % Sensivity
results_final(3,4) = TN/(TN+FP); % Specificity
results_final(3,5) = TP/(TP+FP); % Precision
results_final(3,6) = (2*TP)/(2*TP+FP+FN); % F1-score

comp_time = toc;
results_final(3,7) = comp_time/60/60; % hours


end

%%
%%%%%%%%%%%%%%%%%%%
%%%%% Results %%%%%
%%%%%%%%%%%%%%%%%%%

%%% Testing set %%%
Model={'Hfr.Ofr';'HOHMM';'MHMM'};
temp=table(Model);
results=array2table(results_final,'VariableNames',{'AUC','Accuracy','Sensivity','Specificity','Precision','F1-score','Comp_time'});
if SenAna_q_d == 0
    filename2=[folder,'Replicates_Seed_',num2str(seed_opt),...
        '/Replicate_',num2str(seed),'.xlsx'];
else
    filename2=[folder,'Replicates/Replicate_',num2str(seed),'.xlsx'];
end

writetable([temp,results],filename2);


end

