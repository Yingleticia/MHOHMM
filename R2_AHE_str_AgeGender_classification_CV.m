% -- Download and Install Tensor Toolbox for Matlab -- %
% -- Installation Instructions -- %
% 1. Unpack the files.
% 2. Rename the root directory of the toolbox from tensor_toolbox_2.5 to tensor_toolbox.
% 3. Start MATLAB.
% 4. Within MATLAB, cd to the tensor_toolbox directory and execute the following commands. 
%       addpath(pwd)            %<-- add the tensor toolbox to the MATLAB path
%       cd met; addpath(pwd)    %<-- also add the met directory
%       savepath                %<-- save for future MATLAB sessions

% -- Likewise, also include the utilities folder in the path. 
% -- For circular/angular data, also install and include the CircStat toolbox in the path.


%%%%%% Case Study: MIMIC-III Matched data %%%%%%
%%%%%% Acute Hypotension Episode prediction
%%%%%% Covariate: Age + Gender
clear;clc;close all;
% dbstop if error

%% Set seed 
NRep = 16; % Number of replications
seed = (1 : NRep) + 0; % Set random seeds


%% Parallel implementation
% parfor irep = 1:NRep
for irep = 1:1  
    HMMExp(seed(irep));
    
end

%% Function for the experiment

function [] = HMMExp(seed)

rng(seed);
RandStream.getGlobalStream;

%% Experiment setting

data_select = 2; % 1: raw data; 2: mean; 3: mean+std

% Number of data included 
allData = 0;
NTrain=20; % 20 for 4 groups
NTest=5; % 5 for 4 groups

% results collection
results_final=nan(3,6+1); 
% AUC & Accuracy & Sensivity & Specificity & Precision & F1-score & computation time; 
% Best_MHOHMM, HOHMM, MHMM (Gap0)
cv_metric_select=2; % 1: AUC; 2: Accuracy


%%% load data %%%

if data_select==1
    load('data_mat_AgeGender/AHE_data.mat','AHE_Raw_Gap0','Label','Group')
    dataGap0=AHE_Raw_Gap0;
    method='Raw';
elseif data_select==2
    load('data_mat_AgeGender/AHE_data.mat','AHE_Mean_Gap0','Label','Group')
    dataGap0=AHE_Mean_Gap0;
    method='Mean';
elseif data_select==3
    load('data_mat_AgeGender/AHE_data.mat','AHE_MeanStd_Gap0','Label','Group')
    dataGap0=AHE_MeanStd_Gap0;
    method='MeanVar';
end
[T,dim]=size(dataGap0{1}); % T X dim for each sequence

% Group for two models
level_N=4; % # of groups
% 1: Young Male
% 2: Young Female
% 3: Elderly Male
% 4: Elderly Female

% AHE
A_Y_M_ID=find((Group==1).*(Label==1)==1); 
A_Y_F_ID=find((Group==2).*(Label==1)==1); 
A_E_M_ID=find((Group==3).*(Label==1)==1); 
A_E_F_ID=find((Group==4).*(Label==1)==1); 

% Non-AHE
NA_Y_M_ID=find((Group==1).*(Label==0)==1); 
NA_Y_F_ID=find((Group==2).*(Label==0)==1); 
NA_E_M_ID=find((Group==3).*(Label==0)==1); 
NA_E_F_ID=find((Group==4).*(Label==0)==1); 


if allData==1
    
else
    % Training units
    % AHE(Model 1)
    KK_A_1=NTrain; % group 1
    KK_A_2=NTrain; % group 2
    KK_A_3=NTrain; % group 3
    KK_A_4=NTrain; % group 4
    KK_A=KK_A_1+KK_A_2+KK_A_3+KK_A_4; % total for AHE
    % Non-AHE(Model 2)
    KK_NA_1=NTrain; % group 1
    KK_NA_2=NTrain; % group 2
    KK_NA_3=NTrain; % group 3
    KK_NA_4=NTrain; % group 4
    KK_NA=KK_NA_1+KK_NA_2+KK_NA_3+KK_NA_4; % total for Non-AHE
    % Testing units
    % AHE(Model 1)
    KK_A_t1=NTest; % group 1
    KK_A_t2=NTest; % group 2
    KK_A_t3=NTest; % group 3
    KK_A_t4=NTest; % group 4
    KK_A_t=KK_A_t1+KK_A_t2+KK_A_t3+KK_A_t4; % total for AHE
    % Non-AHE(Model 2)
    KK_NA_t1=NTest; % group 1
    KK_NA_t2=NTest; % group 2
    KK_NA_t3=NTest; % group 3
    KK_NA_t4=NTest; % group 4
    KK_NA_t=KK_NA_t1+KK_NA_t2+KK_NA_t3+KK_NA_t4; % total for Non-AHE
    
    %%%  datasets %%%
    %%% Training %%%
    KK_level1=[ones(1,NTrain),2*ones(1,NTrain),3*ones(1,NTrain),4*ones(1,NTrain)];
    KK_level2=KK_level1; % group index for all units
    T_A_K=repmat(T,KK_A,1);
    T_NA_K=repmat(T,KK_NA,1);
    
    select_A_Y_M=randsample(A_Y_M_ID,KK_A_1);
    select_A_Y_F=randsample(A_Y_F_ID,KK_A_2);
    select_A_E_M=randsample(A_E_M_ID,KK_A_3);
    select_A_E_F=randsample(A_E_F_ID,KK_A_4);
    ID_A_all=[select_A_Y_M;select_A_Y_F;select_A_E_M;select_A_E_F];
    
    select_NA_Y_M=randsample(NA_Y_M_ID,KK_NA_1);
    select_NA_Y_F=randsample(NA_Y_F_ID,KK_NA_2);
    select_NA_E_M=randsample(NA_E_M_ID,KK_NA_3);
    select_NA_E_F=randsample(NA_E_F_ID,KK_NA_4);
    ID_NA_all=[select_NA_Y_M;select_NA_Y_F;select_NA_E_M;select_NA_E_F];
    
    % observation sequences
    Y_A_K_Gap0=dataGap0(ID_A_all); 
    Y_NA_K_Gap0=dataGap0(ID_NA_all); 
    % total sequence
    Y_total_A_Gap0=zeros(T*KK_A,dim);
    Y_total_NA_Gap0=zeros(T*KK_NA,dim);
    for kk=1:KK_A
        Y_total_A_Gap0(((kk-1)*T+1):(kk*T),:)=Y_A_K_Gap0{kk};
        Y_total_NA_Gap0(((kk-1)*T+1):(kk*T),:)=Y_NA_K_Gap0{kk};
    end
    
    %%% Testing %%%
    KK_level_t1=[ones(1,NTest),2*ones(1,NTest),3*ones(1,NTest),4*ones(1,NTest)];
    KK_level_t2=KK_level_t1; 
    KK_level_t=[KK_level_t1,KK_level_t2];
    
    select_A_Y_M_t=randsample(setdiff(A_Y_M_ID,select_A_Y_M),KK_A_t1);
    select_A_Y_F_t=randsample(setdiff(A_Y_F_ID,select_A_Y_F),KK_A_t2);
    select_A_E_M_t=randsample(setdiff(A_E_M_ID,select_A_E_M),KK_A_t3);
    select_A_E_F_t=randsample(setdiff(A_E_F_ID,select_A_E_F),KK_A_t4);
    ID_A_all_t=[select_A_Y_M_t;select_A_Y_F_t;select_A_E_M_t;select_A_E_F_t];   

    select_NA_Y_M_t=randsample(setdiff(NA_Y_M_ID,select_NA_Y_M),KK_NA_t1);
    select_NA_Y_F_t=randsample(setdiff(NA_Y_F_ID,select_NA_Y_F),KK_NA_t2);
    select_NA_E_M_t=randsample(setdiff(NA_E_M_ID,select_NA_E_M),KK_NA_t3);
    select_NA_E_F_t=randsample(setdiff(NA_E_F_ID,select_NA_E_F),KK_NA_t4);
    ID_NA_all_t=[select_NA_Y_M_t;select_NA_Y_F_t;select_NA_E_M_t;select_NA_E_F_t];
 
    ID_all_t=[ID_A_all_t;ID_NA_all_t];
    Y_raw_t_Gap0=dataGap0(ID_all_t); 
end


%% Model Training

%%% Model structure %%%
model_str_all=[1,0,0,0,1;... % MHOHMM: Hf
               1,1,0,0,1;... % MHOHMM: Hfr
               1,0,1,0,1;... % MHOHMM: Hf.Of
               1,1,1,0,1;... % MHOHMM: Hfr.Of
               1,0,0,1,1;... % MHOHMM: Hf.Or
               1,1,0,1,1;... % MHOHMM: Hfr.Or
               1,0,1,1,1;... % MHOHMM: Hf.Ofr
               1,1,1,1,1];   % MHOHMM: Hfr.Ofr
                   % 1 -- fixed effect in hidden process;
                   % 2 -- random effect in hidden process;
                   % 3 -- fixed effect in observed process; 
                   % 4 -- random effect in observed process;
                   % 5 -- higher order in the hidden process
Str_num=size(model_str_all,1);


%%%%%%%%%%%%%%%%%%%%%%%
%%% MHOHMM learning %%%
%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% Parameters %%%%%%%%%%%%
% Stage 1 %
dmax=5;	% maximum number of states
qmax=5;	% maximum dependence order
pM=zeros(qmax,dmax);
for j=1:qmax
    pM(j,1:dmax)=exp(-(j*(1:dmax)/2-1)); % prior probability for k_{j}
    pM(j,1:dmax)=pM(j,1:dmax)./sum(pM(j,1:dmax));
end
alpha00=1;     % prior for Dirichlet distribution for lambda00
alpha_x=1;     % prior for Dirichlet distribution for lambda_x
N1=500;     % number of iterations for first stage

% Stage 2 %
% iteration
simsize=2000; % number of iterations for second stage
burnin=simsize/2;  %floor(simsize/5);
gap=5;
% pi
pigamma(1:qmax)=1./(1*dmax);% prior for Dirichlet distribution for pi
% lambda
alpha_x0=ones(level_N,1); % random effect in lambda
alpha_x1=ones(level_N,1); % fixed effect in lambda
alpha_a0=1; % prior parameters
alpha_b0=1;
alpha_a1=1;
alpha_b1=1;
% weights and psi
omega=[0.3,0.7];
omega_a=1; % prior parameters
omega_b=1;


%%%%%%%%%% Training and testing sets %%%%%%%%%%%%
%%% k-fold cross validation %%%
CV_num=5;

Train_Num_N=KK_A; % 80: 20+20+20+20
N_cv=Train_Num_N/CV_num; % 16
N_group_cv=N_cv/level_N; % 4

cv_metric0=zeros(Str_num,CV_num+1+1); % CV1,...,CV5,Mean,Computation time


%% iteration for k-fold CV
for irep = 1:Str_num
tic

model_str=model_str_all(irep,:);

for ii=1:CV_num
    
% data preparation

Num_cv=N_cv*(CV_num-1); % 16 * 4 = 64
level_cv=[ones(1,Num_cv/level_N),repmat(2,1,Num_cv/level_N),...
    repmat(3,1,Num_cv/level_N),repmat(4,1,Num_cv/level_N)];
level_cv_t=[ones(1,N_group_cv),repmat(2,1,N_group_cv),...
    repmat(3,1,N_group_cv),repmat(4,1,N_group_cv)];
level_cv_t=repmat(level_cv_t,1,2);

% Model AHE
tempTest=((ii-1)*N_group_cv+1):(ii*N_group_cv);
tempTrain=setdiff(1:(Train_Num_N/level_N),tempTest);
Train_cv_idx=[select_A_Y_M(tempTrain);select_A_Y_F(tempTrain);...
    select_A_E_M(tempTrain);select_A_E_F(tempTrain)];
Test_cv_idx=[select_A_Y_M(tempTest);select_A_Y_F(tempTest);...
    select_A_E_M(tempTest);select_A_E_F(tempTest)];
Y_Train_N_cv0=dataGap0(Train_cv_idx);
Y_Test_N_cv0=dataGap0(Test_cv_idx);

Y_total_N_cv0=zeros(Num_cv*T,dim);
for aa=1:Num_cv
    Y_total_N_cv0((T*(aa-1)+1):(T*aa),:)=Y_Train_N_cv0{aa};
end
T_N_cv=repmat(T,Num_cv,1);

% Model Non-AHE
Train_cv_idx=[select_NA_Y_M(tempTrain);select_NA_Y_F(tempTrain);...
    select_NA_E_M(tempTrain);select_NA_E_F(tempTrain)];
Test_cv_idx=[select_NA_Y_M(tempTest);select_NA_Y_F(tempTest);...
    select_NA_E_M(tempTest);select_NA_E_F(tempTest)];
Y_Train_NA_cv0=dataGap0(Train_cv_idx);
Y_Test_NA_cv0=dataGap0(Test_cv_idx);

Y_total_NA_cv0=zeros(Num_cv*T,dim);
for aa=1:Num_cv
    Y_total_NA_cv0((T*(aa-1)+1):(T*aa),:)=Y_Train_NA_cv0{aa};
end
T_NA_cv=repmat(T,Num_cv,1);

% emission distribution
d0init=3;  

if dim==1
    likelihood_type='Normal';
    
    %%%%%%%%%%%%%%%
    %%%   AHE   %%%
    %%%%%%%%%%%%%%%
    
    %%% Gap 0 %%%
    [idx_A_Gap0, MuY_A, sumd]=kmeans(Y_total_N_cv0,d0init);
    tbl=tabulate(idx_A_Gap0);
    SigmaSqY=sumd./(2.*tbl(:,2));
    MuY_A((d0init+1):dmax)=(min(Y_total_N_cv0):(max(Y_total_N_cv0)-min(Y_total_N_cv0))/...
        (dmax-d0init-1):max(Y_total_N_cv0));
    SigmaSqY((d0init+1):dmax)=var(Y_total_N_cv0);
    SigmaSqY(SigmaSqY==0)=var(Y_total_N_cv0);
    Parameters=cell(1,2);
    Parameters{1}=MuY_A;
    Parameters{2}=SigmaSqY;
    Parameters_A_Gap0=Parameters;
    % prior
    Parameters_prior=cell(1,8);
    Parameters_prior{1}=mean(Y_total_N_cv0); % Mu0
    Parameters_prior{2}=3*var(Y_total_N_cv0); % SigmaSq0
    Parameters_prior{3}=1; % kappa0
    Parameters_prior{4}=1; % beta0
    Parameters_prior{5}=0.1*mean(Y_total_N_cv0); % Mu_f
    Parameters_prior{6}=0.1*var(Y_total_N_cv0); % SigmaSq_f
    Parameters_prior{7}=1; % kappa_r
    Parameters_prior{8}=1; % beta_r
    Parameters_prior_A_Gap0=Parameters_prior;
    
    
    %%%%%%%%%%%%%%%%%%%
    %%%   Non-AHE   %%%
    %%%%%%%%%%%%%%%%%%%
    
    %%% Gap 0 %%%
    [idx_NA_Gap0, MuY_NA, sumd]=kmeans(Y_total_NA_cv0,d0init);
    tbl=tabulate(idx_NA_Gap0);
    SigmaSqY=sumd./(2.*tbl(:,2));
    MuY_NA((d0init+1):dmax)=(min(Y_total_NA_cv0):(max(Y_total_NA_cv0)-min(Y_total_NA_cv0))/...
        (dmax-d0init-1):max(Y_total_NA_cv0));
    SigmaSqY((d0init+1):dmax)=var(Y_total_NA_cv0);
    SigmaSqY(SigmaSqY==0)=var(Y_total_NA_cv0);
    Parameters=cell(1,2);
    Parameters{1}=MuY_NA;
    Parameters{2}=SigmaSqY;
    Parameters_NA_Gap0=Parameters;
    % prior
    Parameters_prior=cell(1,8);
    Parameters_prior{1}=mean(Y_total_NA_cv0); % Mu0
    Parameters_prior{2}=3*var(Y_total_NA_cv0); % SigmaSq0
    Parameters_prior{3}=1; % kappa0
    Parameters_prior{4}=1; % beta0
    Parameters_prior{5}=0.1*mean(Y_total_NA_cv0); % Mu_f
    Parameters_prior{6}=0.1*var(Y_total_NA_cv0); % SigmaSq_f
    Parameters_prior{7}=1; % kappa_r
    Parameters_prior{8}=1; % beta_r
    Parameters_prior_NA_Gap0=Parameters_prior;
    
    
elseif dim>1
    likelihood_type='MVN';
    
    %%%%%%%%%%%%%%%
    %%%   AHE   %%%
    %%%%%%%%%%%%%%%
    
    %%% Gap 0 %%%
    [idx_A_Gap0, MuY_A, ~]=kmeans(Y_total_N_cv0,d0init);
    SigmaY_A=zeros(dim,dim,dmax);
    for i=1:dmax
        if (i<=d0init) && (sum(idx_A_Gap0==i)>2)
            SigmaY_A(:,:,i)=cov(Y_total_N_cv0(idx_A_Gap0==i,:));
        else
            SigmaY_A(:,:,i)=cov(Y_total_N_cv0);
        end
    end
    for i=1:dim
        MuY_A((d0init+1):dmax,i)=(min(Y_total_N_cv0(:,i)):...
            (max(Y_total_N_cv0(:,i))-min(Y_total_N_cv0(:,i)))/(dmax-d0init-1)...
            :max(Y_total_N_cv0(:,i)));
    end
    Parameters=cell(1,2);
    Parameters{1}=MuY_A;
    Parameters{2}=SigmaY_A;
    Parameters_A_Gap0=Parameters;
    % prior
    Parameters_prior=cell(1,8);
    Parameters_prior{1}=mean(Y_total_N_cv0); % Mu0
    Parameters_prior{2}=cov(Y_total_N_cv0);% Sigma0
    Parameters_prior{3}=1; % xi0
    Parameters_prior{4}=dim+2; % nu0
    Parameters_prior{5}=0.1*mean(Y_total_N_cv0); % Mu_f
    Parameters_prior{6}=0.1*cov(Y_total_N_cv0); % Sigma_f
    Parameters_prior{7}=eye(dim)*0.01; % Sigma0_r
    Parameters_prior{8}=dim+2; % nu_r
    Parameters_prior_A_Gap0=Parameters_prior;
    
    
    %%%%%%%%%%%%%%%%%%%
    %%%   Non-AHE   %%%
    %%%%%%%%%%%%%%%%%%%
    
    %%% Gap 0 %%%
    [idx_NA_Gap0, MuY_NA, ~]=kmeans(Y_total_NA_cv0,d0init);
    SigmaY_NA=zeros(dim,dim,dmax);
    for i=1:dmax
        if (i<=d0init) && (sum(idx_NA_Gap0==i)>2)
            SigmaY_NA(:,:,i)=cov(Y_total_NA_cv0(idx_NA_Gap0==i,:));
        else
            SigmaY_NA(:,:,i)=cov(Y_total_NA_cv0);
        end
    end
    for i=1:dim
        MuY_NA((d0init+1):dmax,i)=(min(Y_total_NA_cv0(:,i)):...
            (max(Y_total_NA_cv0(:,i))-min(Y_total_NA_cv0(:,i)))/(dmax-d0init-1)...
            :max(Y_total_NA_cv0(:,i)));
    end
    Parameters=cell(1,2);
    Parameters{1}=MuY_NA;
    Parameters{2}=SigmaY_NA;
    Parameters_NA_Gap0=Parameters;
    % prior
    Parameters_prior=cell(1,8);
    Parameters_prior{1}=mean(Y_total_NA_cv0); % Mu0
    Parameters_prior{2}=cov(Y_total_NA_cv0);% Sigma0
    Parameters_prior{3}=1; % xi0
    Parameters_prior{4}=dim+2; % nu0
    Parameters_prior{5}=0.1*mean(Y_total_NA_cv0); % Mu_f
    Parameters_prior{6}=0.1*cov(Y_total_NA_cv0); % Sigma_f
    Parameters_prior{7}=eye(dim)*0.01; % Sigma0_r
    Parameters_prior{8}=dim+2; % nu_r
    Parameters_prior_NA_Gap0=Parameters_prior;
    
end

%%  Gap 0

%%%%%%%%%%%%%%%
%%%   AHE   %%%
%%%%%%%%%%%%%%%
beta=0.5; % initial fixed effect
C_total=idx_A_Gap0;
CC_K=cell(1,Num_cv); % initial state sequences 
for aa=1:Num_cv
    CC_K{aa}=C_total((T*(aa-1)+1):(T*aa));
end

%%% Sampler for Stage 1
disp(['------------------ CV ',num2str(ii),' ------------------'])
fprintf('\n--- MHOHMM (Model AHE; Gap0) --- \n\n');
fprintf('\n--- First Stage --- \n\n');
[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Num_cv,level_N,level_cv,CC_K,Y_Train_N_cv0,...
    Y_total_N_cv0,T_N_cv,alpha_x,alpha00,Parameters_A_Gap0,...
    beta,likelihood_type,Parameters_prior_A_Gap0,pM,model_str);

%%% Sampler for Stage 2
fprintf('\n --- Second Stage --- \n\n');
pause(1);
[lambda_x_Mix_N,lambda_x0_Mix_N,alpha_x0_Mix_N,alpha_x1_Mix_N,omega_Mix_N,...
    pi_Mix_N,mu_Mix_N,sigma_Mix_N,beta_Mix_N,Sigma_r_Mix_N,L00_Mix_N,...
    State_select_Mix_N,p00_Mix_N,K00_Mix_N,ind00_Mix_N,likeli_Mix_N]=...
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Num_cv,...
    level_N,level_cv,C_total,CC_K,Xnew,Cnew,Y_Train_N_cv0,Y_total_N_cv0,...
    T_N_cv,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,....
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_A_Gap0,model_str);

%%%%%%%%%%%%%%%%%%%
%%%   Non-AHE   %%%
%%%%%%%%%%%%%%%%%%%

beta=0.5; % initial fixed effect
C_total=idx_NA_Gap0;
CC_K=cell(1,Num_cv); % initial state sequences 
for aa=1:Num_cv
    CC_K{aa}=C_total((T*(aa-1)+1):(T*aa));
end

%%% Sampler for Stage 1
fprintf('\n--- MHOHMM (Model Non-AHE; Gap0) --- \n\n');
fprintf('\n--- First Stage --- \n\n');
[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Num_cv,level_N,level_cv,CC_K,Y_Train_NA_cv0,...
    Y_total_NA_cv0,T_NA_cv,alpha_x,alpha00,Parameters_NA_Gap0,...
    beta,likelihood_type,Parameters_prior_NA_Gap0,pM,model_str);

%%% Sampler for Stage 2
fprintf('\n --- Second Stage --- \n\n');
pause(1);
[lambda_x_Mix_NA,lambda_x0_Mix_NA,alpha_x0_Mix_NA,alpha_x1_Mix_NA,omega_Mix_NA,pi_Mix_NA,...
    mu_Mix_NA,sigma_Mix_NA,beta_Mix_NA,Sigma_r_Mix_NA,L00_Mix_NA,...
    State_select_Mix_NA,p00_Mix_NA,K00_Mix_NA,ind00_Mix_NA,likeli_Mix_NA]=...
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Num_cv,...
    level_N,level_cv,C_total,CC_K,Xnew,Cnew,Y_Train_NA_cv0,Y_total_NA_cv0,...
    T_NA_cv,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,....
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_NA_Gap0,model_str);

%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Classification  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
sampsize=1000;
num_para=3;
KK_t=length(level_cv_t); % num of testing units
Y_Test_cv=[Y_Test_N_cv0,Y_Test_NA_cv0]; % data for testing
classTrue=[ones(1,N_cv),repmat(2,1,N_cv)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Classification MHOHMM %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class_test_MHOHMM=ones(1,KK_t); 
scoreN_MHOHMM=ones(1,KK_t); 
scoreNA_MHOHMM=ones(1,KK_t);
disp('--- MHOHMM Classification ---');  
for kk=1:KK_t
    group=level_cv_t(kk);
    data=Y_Test_cv{kk};

    % Model N %
    [LogLikeli_N,~,~]=decode_MHOHMM_str_aoas(data,group,lambda_x_Mix_N,...
        lambda_x0_Mix_N,alpha_x0_Mix_N,omega_Mix_N,pi_Mix_N,mu_Mix_N,...
        sigma_Mix_N,beta_Mix_N,Sigma_r_Mix_N,likelihood_type,qmax,dmax,...
        sampsize,p00_Mix_N,K00_Mix_N,ind00_Mix_N,num_para,model_str);
    % Model NA %
    [LogLikeli_NA,~,~]=decode_MHOHMM_str_aoas(data,group,lambda_x_Mix_NA,...
        lambda_x0_Mix_NA,alpha_x0_Mix_NA,omega_Mix_NA,pi_Mix_NA,mu_Mix_NA,...
        sigma_Mix_NA,beta_Mix_NA,Sigma_r_Mix_NA,likelihood_type,qmax,dmax,...
        sampsize,p00_Mix_NA,K00_Mix_NA,ind00_Mix_NA,num_para,model_str);

    if mean(LogLikeli_N)<mean(LogLikeli_NA)
        class_test_MHOHMM(kk)=2; % NA class
    end
    scoreN_MHOHMM(kk)=mean(LogLikeli_N);
    scoreNA_MHOHMM(kk)=mean(LogLikeli_NA);
end
[~,~,~,auc_M]=perfcurve(classTrue,(scoreN_MHOHMM-scoreNA_MHOHMM),1); % MHOHMM
if cv_metric_select==1 % AUC
   cv_metric0(irep,ii)=auc_M; 
elseif cv_metric_select==2 % Accuracy
   cv_metric0(irep,ii)=sum(class_test_MHOHMM==classTrue)/KK_t; 
end

end

toc
cv_metric0(irep,end)=toc/60/60; % hours
end

cv_metric_mean=mean(cv_metric0(:,1:CV_num),2);
cv_metric0(:,end-1)=cv_metric_mean;
[~,str_id0]=max(cv_metric_mean);


%%% Cross validation set %%%
Model_Name={'Hf(Gap0)';'Hfr(Gap0)';'Hf.Of(Gap0)';'Hfr.Of(Gap0)';...
    'Hf.Or(Gap0)';'Hfr.Or(Gap0)';'Hf.Ofr(Gap0)';'Hfr.Ofr(Gap0)'}; 
temp=table(Model_Name);
Vname=cell(1,CV_num+1+1);
for cc=1:CV_num
    tempcv=['CV',num2str(cc)];
    Vname{cc}=tempcv;
end
Vname{end-1}='Mean';
Vname{end}='Time(hrs)';

cv_metric=cv_metric0;
cv_metric=array2table(cv_metric,'VariableNames',Vname);

if cv_metric_select==1 % AUC
    filename1=['Result_AgeGender/CV/',method,'_AUC_CV',num2str(CV_num),'_Seed_',num2str(seed),'.xlsx'];
elseif cv_metric_select==2 % Accuracy
    filename1=['Result_AgeGender/CV/',method,'_Accuracy_CV',num2str(CV_num),'_Seed_',num2str(seed),'.xlsx'];
end
writetable([temp,cv_metric],filename1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Final Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if str_id0==str_id30 % same best structure
if 1==1
    
% emission distribution
if dim==1
    
    %%%%%%%%%%%%%%%
    %%%   AHE   %%%
    %%%%%%%%%%%%%%%
    
    %%% Gap 0 %%%
    [idx_A_Gap0, MuY_A, sumd]=kmeans(Y_total_A_Gap0,d0init);
    tbl=tabulate(idx_A_Gap0);
    SigmaSqY=sumd./(2.*tbl(:,2));
    MuY_A((d0init+1):dmax)=(min(Y_total_A_Gap0):(max(Y_total_A_Gap0)-min(Y_total_A_Gap0))/...
        (dmax-d0init-1):max(Y_total_A_Gap0));
    SigmaSqY((d0init+1):dmax)=var(Y_total_A_Gap0);
    SigmaSqY(SigmaSqY==0)=var(Y_total_A_Gap0);
    Parameters=cell(1,2);
    Parameters{1}=MuY_A;
    Parameters{2}=SigmaSqY;
    Parameters_A_Gap0=Parameters;
    % prior
    Parameters_prior=cell(1,8);
    Parameters_prior{1}=mean(Y_total_A_Gap0); % Mu0
    Parameters_prior{2}=3*var(Y_total_A_Gap0); % SigmaSq0
    Parameters_prior{3}=1; % kappa0
    Parameters_prior{4}=1; % beta0
    Parameters_prior{5}=0.1*mean(Y_total_A_Gap0); % Mu_f
    Parameters_prior{6}=0.1*var(Y_total_A_Gap0); % SigmaSq_f
    Parameters_prior{7}=1; % kappa_r
    Parameters_prior{8}=1; % beta_r
    Parameters_prior_A_Gap0=Parameters_prior;
   
    
    
    %%%%%%%%%%%%%%%%%%%
    %%%   Non-AHE   %%%
    %%%%%%%%%%%%%%%%%%%
    
    %%% Gap 0 %%%
    [idx_NA_Gap0, MuY_NA, sumd]=kmeans(Y_total_NA_Gap0,d0init);
    tbl=tabulate(idx_NA_Gap0);
    SigmaSqY=sumd./(2.*tbl(:,2));
    MuY_NA((d0init+1):dmax)=(min(Y_total_NA_Gap0):(max(Y_total_NA_Gap0)-min(Y_total_NA_Gap0))/...
        (dmax-d0init-1):max(Y_total_NA_Gap0));
    SigmaSqY((d0init+1):dmax)=var(Y_total_NA_Gap0);
    SigmaSqY(SigmaSqY==0)=var(Y_total_NA_Gap0);
    Parameters=cell(1,2);
    Parameters{1}=MuY_NA;
    Parameters{2}=SigmaSqY;
    Parameters_NA_Gap0=Parameters;
    % prior
    Parameters_prior=cell(1,8);
    Parameters_prior{1}=mean(Y_total_NA_Gap0); % Mu0
    Parameters_prior{2}=3*var(Y_total_NA_Gap0); % SigmaSq0
    Parameters_prior{3}=1; % kappa0
    Parameters_prior{4}=1; % beta0
    Parameters_prior{5}=0.1*mean(Y_total_NA_Gap0); % Mu_f
    Parameters_prior{6}=0.1*var(Y_total_NA_Gap0); % SigmaSq_f
    Parameters_prior{7}=1; % kappa_r
    Parameters_prior{8}=1; % beta_r
    Parameters_prior_NA_Gap0=Parameters_prior;
    
    
elseif dim>1
    
    %%%%%%%%%%%%%%%
    %%%   AHE   %%%
    %%%%%%%%%%%%%%%
    
    %%% Gap 0 %%%
    [idx_A_Gap0, MuY_A, ~]=kmeans(Y_total_A_Gap0,d0init);
    SigmaY_A=zeros(dim,dim,dmax);
    for i=1:dmax
        if (i<=d0init) && (sum(idx_A_Gap0==i)>2)
            SigmaY_A(:,:,i)=cov(Y_total_A_Gap0(idx_A_Gap0==i,:));
        else
            SigmaY_A(:,:,i)=cov(Y_total_A_Gap0);
        end
    end
    for i=1:dim
        MuY_A((d0init+1):dmax,i)=(min(Y_total_A_Gap0(:,i)):...
            (max(Y_total_A_Gap0(:,i))-min(Y_total_A_Gap0(:,i)))/(dmax-d0init-1)...
            :max(Y_total_A_Gap0(:,i)));
    end
    Parameters=cell(1,2);
    Parameters{1}=MuY_A;
    Parameters{2}=SigmaY_A;
    Parameters_A_Gap0=Parameters;
    % prior
    Parameters_prior=cell(1,8);
    Parameters_prior{1}=mean(Y_total_A_Gap0); % Mu0
    Parameters_prior{2}=cov(Y_total_A_Gap0);% Sigma0
    Parameters_prior{3}=1; % xi0
    Parameters_prior{4}=dim+2; % nu0
    Parameters_prior{5}=0.1*mean(Y_total_A_Gap0); % Mu_f
    Parameters_prior{6}=0.1*cov(Y_total_A_Gap0); % Sigma_f
    Parameters_prior{7}=eye(dim)*0.01; % Sigma0_r
    Parameters_prior{8}=dim+2; % nu_r
    Parameters_prior_A_Gap0=Parameters_prior;
   
    
    %%%%%%%%%%%%%%%%%%%
    %%%   Non-AHE   %%%
    %%%%%%%%%%%%%%%%%%%
    
    %%% Gap 0 %%%
    [idx_NA_Gap0, MuY_NA, ~]=kmeans(Y_total_NA_Gap0,d0init);
    SigmaY_NA=zeros(dim,dim,dmax);
    for i=1:dmax
        if (i<=d0init) && (sum(idx_NA_Gap0==i)>2)
            SigmaY_NA(:,:,i)=cov(Y_total_NA_Gap0(idx_NA_Gap0==i,:));
        else
            SigmaY_NA(:,:,i)=cov(Y_total_NA_Gap0);
        end
    end
    for i=1:dim
        MuY_NA((d0init+1):dmax,i)=(min(Y_total_NA_Gap0(:,i)):...
            (max(Y_total_NA_Gap0(:,i))-min(Y_total_NA_Gap0(:,i)))/(dmax-d0init-1)...
            :max(Y_total_NA_Gap0(:,i)));
    end
    Parameters=cell(1,2);
    Parameters{1}=MuY_NA;
    Parameters{2}=SigmaY_NA;
    Parameters_NA_Gap0=Parameters;
    % prior
    Parameters_prior=cell(1,8);
    Parameters_prior{1}=mean(Y_total_NA_Gap0); % Mu0
    Parameters_prior{2}=cov(Y_total_NA_Gap0);% Sigma0
    Parameters_prior{3}=1; % xi0
    Parameters_prior{4}=dim+2; % nu0
    Parameters_prior{5}=0.1*mean(Y_total_NA_Gap0); % Mu_f
    Parameters_prior{6}=0.1*cov(Y_total_NA_Gap0); % Sigma_f
    Parameters_prior{7}=eye(dim)*0.01; % Sigma0_r
    Parameters_prior{8}=dim+2; % nu_r
    Parameters_prior_NA_Gap0=Parameters_prior;
    
end

model_str_best0=model_str_all(str_id0,:);

%%% save data for replications of final models %%%
filename0=['Result_AgeGender/CV/',method,'_Final_data_Seed_',num2str(seed),'.mat'];
save(filename0,'str_id0','Str_num','model_str_best0',...
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


%%  Gap 0
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

toc
results_final(1,7) = toc/60/60; %hours
%%
%%%%%%%%%%%%%%%%%%%%%%
%%% HOHMM learning %%%
%%%%%%%%%%%%%%%%%%%%%%
tic
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
toc
results_final(2,7) = toc/60/60; %hours

%%
%%%%%%%%%%%%%%%%%%%%%
%%% MHMM learning %%%
%%%%%%%%%%%%%%%%%%%%%
tic
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
toc
results_final(3,7) = toc/60/60; %hours

%%
%%%%%%%%%%%%%%%%%%%
%%%%% Results %%%%%
%%%%%%%%%%%%%%%%%%%
Model_Name={'Hf(Gap0)';'Hfr(Gap0)';'Hf.Of(Gap0)';'Hfr.Of(Gap0)';...
    'Hf.Or(Gap0)';'Hfr.Or(Gap0)';'Hf.Ofr(Gap0)';'Hfr.Ofr(Gap0)'};

%%% Testing set %%%
Model={Model_Name{str_id0};'HOHMM(Gap0)';'MHMM(Gap0)'};
temp=table(Model);
results=array2table(results_final,'VariableNames',{'AUC','Accuracy',...
    'Sensivity','Specificity','Precision','F1-score','Time(hrs)'});
filename2=['Result_AgeGender/CV/',method,'_Final_classification_test_Seed_',num2str(seed),'.xlsx'];
writetable([temp,results],filename2);


end


end