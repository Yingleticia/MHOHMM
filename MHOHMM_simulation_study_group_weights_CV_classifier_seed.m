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
NRep = 16; % Number of replications
seed = (1 : NRep) + 0; % Set random seeds

compute_time=zeros(1,NRep);


% dbstop if error;

%% Parallel implementation
parfor ss = 1:NRep
% for irep = 1:NRep  
    temp=HMMExp(seed(ss));
    compute_time(ss)=temp;
end
disp('-------  Computation Time (hour) -------')
disp(compute_time/60/60)


%%
function [compute_time] = HMMExp(seed)

tic
rng(seed);
RandStream.getGlobalStream;

%%%%%%%%%%%%%%%%%%%%%
%%% Simulate Data %%%
%%%%%%%%%%%%%%%%%%%%%

% Model parameters setting

%%% Model structure %%%
% for data generating
model_str_0=[1,1,1,1,1]; 
                   % 1 -- fixed effect in hidden process;
                   % 2 -- random effect in hidden process;
                   % 3 -- fixed effect in observed process; 
                   % 4 -- random effect in observed process;
                   % 5 -- higher order in the hidden process
%%%%%%%%%%%%%%%%%%%%
% for model training  
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
                   
%%%%%%%%%%%%%%%%%%%
classification=1; % do classification or not

KK=100; % number of sequences for each model
T=200; % number of each sequence

% emission distributions
mu0_N=[0,2,4]; % model 1: 'Normal'
mu0_NA=[0,2,4]; % model 2: 'Abnormal'
beta_N0=1;
beta_NA0=1;
sigma0_R=0.5;

% weights w_1
omega1_1=0.6; % group 1
omega1_2=0.8; % group 2

% alpha
alpha0_0=1; % for random effect
alpha1_0=1; % for fixed effect

% order
% model 1: 'Normal' 
q1_A=2;
q2_A=3;
% model 2: 'Abnormal'
q1_NA=2;
q2_NA=4;

%% results collection
results_final=nan(3,6); 
% AUC & Accuracy & Sensivity & Specificity & Precision & F1-score; 
% Best_MHOHMM, HOHMM, MHMM
cv_metric_select=2; % 1: AUC; 2: Accuracy
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
level_N=2; % # of groups
KK_level=ones(1,KK); % group index for all units
KK_level((KK/2+1):end)=2; % 1: group A; 2: group B
d_0=3; %number of states 

%%% emission %%% 
likelihood_type='Normal';
sigma0_N=0.5;
sigma0_NA=0.5;
if model_str_0(3)==0 % No fixed effect in observed process
    beta_N0=0;
    beta_NA0=0;
end
if model_str_0(4)==0 % No random effect in observed process
    sigma0_R=0;
end

%%% transition %%% 
if model_str_0(2)==0 % No random effect in hidden process
    omega1_1=1;
    omega1_2=1;
end 
omega0_1=1-omega1_1;  
omega0_2=1-omega1_2; 

% model 1: 'Normal' Order--1,2
% state preference

if 1==1
    % random generate (method 2)
    lambda0_N_A=zeros(d_0,1);
    lambda0_N_A(1)=betarnd(1,1,1);
    lambda0_N_A(2)=betarnd(1,1,1)*(1-lambda0_N_A(1));
    lambda0_N_A(3)=1-lambda0_N_A(1)-lambda0_N_A(2);
    
    while(sum(lambda0_N_A<0.1)>0) % in case the probability is so small
        lambda0_N_A=zeros(d_0,1);
        lambda0_N_A(1)=betarnd(1,1,1);
        lambda0_N_A(2)=betarnd(1,1,1)*(1-lambda0_N_A(1));
        lambda0_N_A(3)=1-lambda0_N_A(1)-lambda0_N_A(2);
    end
    
    lambda0_N_B=zeros(d_0,1);
    lambda0_N_B(1)=betarnd(1,1,1);
    lambda0_N_B(2)=betarnd(1,1,1)*(1-lambda0_N_B(1));
    lambda0_N_B(3)=1-lambda0_N_B(1)-lambda0_N_B(2);
    
    while(sum(lambda0_N_B<0.1)>0) % in case the probability is so small
        lambda0_N_B=zeros(d_0,1);
        lambda0_N_B(1)=betarnd(1,1,1);
        lambda0_N_B(2)=betarnd(1,1,1)*(1-lambda0_N_B(1));
        lambda0_N_B(3)=1-lambda0_N_B(1)-lambda0_N_B(2);
    end
end

% group-level transition probability
if 1==1
    lambda_x_N_A=tensor(zeros(repmat(d_0,1,q1_A+1)),repmat(d_0,1,q1_A+1));
    if q1_A==1
        for j=1:d_0
            temp=gamrnd(alpha1_0*lambda0_N_A,1);
            temp=temp/sum(temp);
            lambda_x_N_A(:,j)=temp;
        end   
    elseif q1_A==2
        for j=1:d_0
            for a=1:d_0
                temp=gamrnd(alpha1_0*lambda0_N_A,1);
                temp=temp/sum(temp);
                lambda_x_N_A(:,j,a)=temp;
            end
        end
    elseif q1_A==3
        for j=1:d_0
            for a=1:d_0
                for b=1:d_0
                    temp=gamrnd(alpha1_0*lambda0_N_A,1);
                    temp=temp/sum(temp);
                    lambda_x_N_A(:,j,a,b)=temp;
                end
            end
        end
    end
end


if 1==1
    lambda_x_N_B=tensor(zeros(repmat(d_0,1,q2_A+1)),repmat(d_0,1,q2_A+1)); 
    if q2_A==2
        for j=1:d_0
            for a=1:d_0
                temp=gamrnd(alpha1_0*lambda0_N_B,1);
                temp=temp/sum(temp);
                lambda_x_N_B(:,j,a)=temp;
            end
        end
    elseif q2_A==3
        for j=1:d_0
            for a=1:d_0
                for b=1:d_0
                    temp=gamrnd(alpha1_0*lambda0_N_B,1);
                    temp=temp/sum(temp);
                    lambda_x_N_B(:,j,a,b)=temp;
                end
            end
        end
    elseif q2_A==4
        for j=1:d_0
            for a=1:d_0
                for b=1:d_0
                    for c=1:d_0
                        temp=gamrnd(alpha1_0*lambda0_N_B,1);
                        temp=temp/sum(temp);
                        lambda_x_N_B(:,j,a,b,c)=temp;
                    end
                end
            end
        end
    end
end

       
% model 2: 'Abnormal' Order--1,3
% state preference

if 1==1
    % random generate (method 2)
    lambda0_NA_A=zeros(d_0,1);
    lambda0_NA_A(1)=betarnd(1,1,1);
    lambda0_NA_A(2)=betarnd(1,1,1)*(1-lambda0_NA_A(1));
    lambda0_NA_A(3)=1-lambda0_NA_A(1)-lambda0_NA_A(2);
    
    while(sum(lambda0_NA_A<0.1)>0) % in case the probability is so small
        lambda0_NA_A=zeros(d_0,1);
        lambda0_NA_A(1)=betarnd(1,1,1);
        lambda0_NA_A(2)=betarnd(1,1,1)*(1-lambda0_NA_A(1));
        lambda0_NA_A(3)=1-lambda0_NA_A(1)-lambda0_NA_A(2);
    end
    
    lambda0_NA_B=zeros(d_0,1);
    lambda0_NA_B(1)=betarnd(1,1,1);
    lambda0_NA_B(2)=betarnd(1,1,1)*(1-lambda0_NA_B(1));
    lambda0_NA_B(3)=1-lambda0_NA_B(1)-lambda0_NA_B(2);
    
    while(sum(lambda0_NA_B<0.1)>0) % in case the probability is so small
        lambda0_NA_B=zeros(d_0,1);
        lambda0_NA_B(1)=betarnd(1,1,1);
        lambda0_NA_B(2)=betarnd(1,1,1)*(1-lambda0_NA_B(1));
        lambda0_NA_B(3)=1-lambda0_NA_B(1)-lambda0_NA_B(2);
    end
end

% group-level transition probability
if 1==1
    lambda_x_NA_A=tensor(zeros(repmat(d_0,1,q1_NA+1)),repmat(d_0,1,q1_NA+1)); 
    if q1_NA==1
        for j=1:d_0
            temp=gamrnd(alpha1_0*lambda0_NA_A,1);
            temp=temp/sum(temp);
            lambda_x_NA_A(:,j)=temp;
        end   
    elseif q1_NA==2
        for j=1:d_0
            for a=1:d_0
                temp=gamrnd(alpha1_0*lambda0_NA_A,1);
                temp=temp/sum(temp);
                lambda_x_NA_A(:,j,a)=temp;
            end
        end
    elseif q1_NA==3
        for j=1:d_0
            for a=1:d_0
                for b=1:d_0
                    temp=gamrnd(alpha1_0*lambda0_NA_A,1);
                    temp=temp/sum(temp);
                    lambda_x_NA_A(:,j,a,b)=temp;
                end
            end
        end
    end
end


if 1==1
    lambda_x_NA_B=tensor(zeros(repmat(d_0,1,q2_NA+1)),repmat(d_0,1,q2_NA+1)); 
    if q2_NA==2
        for j=1:d_0
            for a=1:d_0
                temp=gamrnd(alpha1_0*lambda0_NA_B,1);
                temp=temp/sum(temp);
                lambda_x_NA_B(:,j,a)=temp;
            end
        end
    elseif q2_NA==3
        for j=1:d_0
            for a=1:d_0
                for b=1:d_0
                    temp=gamrnd(alpha1_0*lambda0_NA_B,1);
                    temp=temp/sum(temp);
                    lambda_x_NA_B(:,j,a,b)=temp;
                end
            end
        end
    elseif q2_NA==4
        for j=1:d_0
            for a=1:d_0
                for b=1:d_0
                    for c=1:d_0
                        temp=gamrnd(alpha1_0*lambda0_NA_B,1);
                        temp=temp/sum(temp);
                        lambda_x_NA_B(:,j,a,b,c)=temp;
                    end
                end
            end
        end
    end
end

if 1==1
    disp(lambda0_N_A);
    disp(lambda0_N_B);
    disp(lambda0_NA_A);
    disp(lambda0_NA_B);
end
%%
%%%%%%%%%%%%%%%
%%% Model N %%%
%%%%%%%%%%%%%%%
Y_N_K=cell(1,KK); % observation sequences
C_N_K=cell(1,KK); % state sequences
% T_N_K=repmat(T,KK,1); % number of observations in each sequence
% C_N_total_true=zeros(KK*T,1); % true states for all sequences

% starting state(s)
% C_start_N=randsample(d_0,3,'true');
for aa=1:KK
   C_0=zeros(T,1); % state sequence
   Y_0=zeros(T,1); % observation sequence
   % unit-specific mu
   mu_aa=normrnd(0,sigma0_R,1); % mu_a
   mu_R=mu0_N+mu_aa+beta_N0*KK_level(aa);
   
   if aa<=(KK/2)
       %%% Group A
       % mixed effect: lambda
       lambda_temp=tensor(zeros(repmat(d_0,1,q1_A+1)),repmat(d_0,1,q1_A+1));
       if q1_A==1
            for j=1:d_0
                temp=gamrnd(alpha0_0*lambda0_N_A,1);
                temp=temp/sum(temp);
                lambda_temp(:,j)=temp;
            end   
            lambda_aa=omega1_1*lambda_x_N_A+omega0_1*lambda_temp;
            C_0(1)=randsample(d_0,1,'true');
            Y_0(1)=normrnd(mu_R(C_0(1)),sigma0_N,1);
            for i=2:T
                C_0(i)=randsample(d_0,1,'true',[lambda_aa(1,C_0(i-1)),...
                         lambda_aa(2,C_0(i-1)),lambda_aa(3,C_0(i-1))]);
                Y_0(i)=normrnd(mu_R(C_0(i)),sigma0_N,1);
            end
       elseif q1_A==2
            for j=1:d_0
                for a=1:d_0
                    temp=gamrnd(alpha0_0*lambda0_N_A,1);
                    temp=temp/sum(temp);
                    lambda_temp(:,j,a)=temp;
                end
            end
            lambda_aa=omega1_1*lambda_x_N_A+omega0_1*lambda_temp;
            C_0(1:2)=randsample(d_0,2,'true');
            Y_0(1)=normrnd(mu_R(C_0(1)),sigma0_N,1);
            Y_0(2)=normrnd(mu_R(C_0(2)),sigma0_N,1);
            for i=3:T
                C_0(i)=randsample(d_0,1,'true',[lambda_aa(1,C_0(i-1),C_0(i-2)),...
                         lambda_aa(2,C_0(i-1),C_0(i-2)),lambda_aa(3,C_0(i-1),C_0(i-2))]);
                Y_0(i)=normrnd(mu_R(C_0(i)),sigma0_N,1);
            end
       elseif q1_A==3
            for j=1:d_0
                for a=1:d_0
                    for b=1:d_0
                        temp=gamrnd(alpha0_0*lambda0_N_A,1);
                        temp=temp/sum(temp);
                        lambda_temp(:,j,a,b)=temp;
                    end
                end
            end
            lambda_aa=omega1_1*lambda_x_N_A+omega0_1*lambda_temp;
            C_0(1:3)=randsample(d_0,3,'true');
            Y_0(1)=normrnd(mu_R(C_0(1)),sigma0_N,1);
            Y_0(2)=normrnd(mu_R(C_0(2)),sigma0_N,1);
            Y_0(3)=normrnd(mu_R(C_0(3)),sigma0_N,1);
            for i=4:T
                C_0(i)=randsample(d_0,1,'true',[lambda_aa(1,C_0(i-1),C_0(i-2),C_0(i-3)),...
                         lambda_aa(2,C_0(i-1),C_0(i-2),C_0(i-3)),lambda_aa(3,C_0(i-1),C_0(i-2),C_0(i-3))]);
                Y_0(i)=normrnd(mu_R(C_0(i)),sigma0_N,1);
            end
       end
   else
       %%% Group B
       % mixed effect: lambda
       lambda_temp=tensor(zeros(repmat(d_0,1,q2_A+1)),repmat(d_0,1,q2_A+1));
       if q2_A==2
            for j=1:d_0
                for a=1:d_0
                    temp=gamrnd(alpha0_0*lambda0_N_B,1);
                    temp=temp/sum(temp);
                    lambda_temp(:,j,a)=temp;
                end
            end
            lambda_aa=omega1_2*lambda_x_N_B+omega0_2*lambda_temp;
            % Generate sequences
            C_0(1:2)=randsample(d_0,2,'true');
            Y_0(1)=normrnd(mu_R(C_0(1)),sigma0_N,1);
            Y_0(2)=normrnd(mu_R(C_0(2)),sigma0_N,1);
            for i=3:T
                C_0(i)=randsample(d_0,1,'true',[lambda_aa(1,C_0(i-1),C_0(i-2)),...
                         lambda_aa(2,C_0(i-1),C_0(i-2)),lambda_aa(3,C_0(i-1),C_0(i-2))]);
                Y_0(i)=normrnd(mu_R(C_0(i)),sigma0_N,1);
            end
        elseif q2_A==3
            for j=1:d_0
                for a=1:d_0
                    for b=1:d_0
                        temp=gamrnd(alpha0_0*lambda0_N_B,1);
                        temp=temp/sum(temp);
                        lambda_temp(:,j,a,b)=temp;
                    end
                end
            end
            lambda_aa=omega1_2*lambda_x_N_B+omega0_2*lambda_temp;
            % Generate sequences
            C_0(1:3)=randsample(d_0,3,'true');
            Y_0(1)=normrnd(mu_R(C_0(1)),sigma0_N,1);
            Y_0(2)=normrnd(mu_R(C_0(2)),sigma0_N,1);
            Y_0(3)=normrnd(mu_R(C_0(3)),sigma0_N,1);
            for i=4:T
                C_0(i)=randsample(d_0,1,'true',[lambda_aa(1,C_0(i-1),C_0(i-2),C_0(i-3)),...
                         lambda_aa(2,C_0(i-1),C_0(i-2),C_0(i-3)),lambda_aa(3,C_0(i-1),C_0(i-2),C_0(i-3))]);
                Y_0(i)=normrnd(mu_R(C_0(i)),sigma0_N,1);
            end
        elseif q2_A==4
            for j=1:d_0
                for a=1:d_0
                    for b=1:d_0
                        for c=1:d_0
                            temp=gamrnd(alpha0_0*lambda0_N_B,1);
                            temp=temp/sum(temp);
                            lambda_temp(:,j,a,b,c)=temp;
                        end
                    end
                end
            end
            lambda_aa=omega1_2*lambda_x_N_B+omega0_2*lambda_temp;
            % Generate sequences
            C_0(1:4)=randsample(d_0,4,'true');
            Y_0(1)=normrnd(mu_R(C_0(1)),sigma0_N,1);
            Y_0(2)=normrnd(mu_R(C_0(2)),sigma0_N,1);
            Y_0(3)=normrnd(mu_R(C_0(3)),sigma0_N,1);
            Y_0(4)=normrnd(mu_R(C_0(4)),sigma0_N,1);
            for i=5:T
                C_0(i)=randsample(d_0,1,'true',[lambda_aa(1,C_0(i-1),C_0(i-2),C_0(i-3),C_0(i-4)),...
                         lambda_aa(2,C_0(i-1),C_0(i-2),C_0(i-3),C_0(i-4)),...
                         lambda_aa(3,C_0(i-1),C_0(i-2),C_0(i-3),C_0(i-4))]);
                Y_0(i)=normrnd(mu_R(C_0(i)),sigma0_N,1);
            end
        end

   end
   Y_N_K{aa}=Y_0;
   C_N_K{aa}=C_0;
%    C_N_total_true((T*(aa-1)+1):(T*aa))=C_N_K{aa};
end

% plot generated sequences
if 1==0
    figure;
    for aa=1:KK
        plot(Y_N_K{aa});hold on;   
    end
end
    
%%
%%%%%%%%%%%%%%%%
%%% Model NA %%%
%%%%%%%%%%%%%%%%
Y_NA_K=cell(1,KK); % observation sequences
C_NA_K=cell(1,KK); % state sequences
% T_NA_K=repmat(T,KK,1); % number of observations in each sequence
% C_NA_total_true=zeros(KK*T,1); % true states for all sequences

% starting state(s)
% C_start_NA=randsample(d_0,3,'true');
for aa=1:KK
   C_0=zeros(T,1); % state sequence
   Y_0=zeros(T,1); % observation sequence
   % unit-specific mu
   mu_aa=normrnd(0,sigma0_R,1); % mu_a
   mu_R=mu0_NA+mu_aa+beta_NA0*KK_level(aa);
   
   if aa<=(KK/2)
       %%% Group A
       % mixed effect: lambda
       lambda_temp=tensor(zeros(repmat(d_0,1,q1_NA+1)),repmat(d_0,1,q1_NA+1));
       if q1_NA==1
            for j=1:d_0
                temp=gamrnd(alpha0_0*lambda0_NA_A,1);
                temp=temp/sum(temp);
                lambda_temp(:,j)=temp;
            end   
            lambda_aa=omega1_1*lambda_x_NA_A+omega0_1*lambda_temp;
            % Generate sequences
            C_0(1)=randsample(d_0,1,'true');
            Y_0(1)=normrnd(mu_R(C_0(1)),sigma0_NA,1);
            for i=2:T
                C_0(i)=randsample(d_0,1,'true',[lambda_aa(1,C_0(i-1)),...
                         lambda_aa(2,C_0(i-1)),lambda_aa(3,C_0(i-1))]);
                Y_0(i)=normrnd(mu_R(C_0(i)),sigma0_NA,1);
            end
        elseif q1_NA==2
            for j=1:d_0
                for a=1:d_0
                    temp=gamrnd(alpha0_0*lambda0_NA_A,1);
                    temp=temp/sum(temp);
                    lambda_temp(:,j,a)=temp;
                end
            end
            lambda_aa=omega1_1*lambda_x_NA_A+omega0_1*lambda_temp;
            C_0(1:2)=randsample(d_0,2,'true');
            Y_0(1)=normrnd(mu_R(C_0(1)),sigma0_NA,1);
            Y_0(2)=normrnd(mu_R(C_0(2)),sigma0_NA,1);
            for i=3:T
                C_0(i)=randsample(d_0,1,'true',[lambda_aa(1,C_0(i-1),C_0(i-2)),...
                         lambda_aa(2,C_0(i-1),C_0(i-2)),lambda_aa(3,C_0(i-1),C_0(i-2))]);
                Y_0(i)=normrnd(mu_R(C_0(i)),sigma0_NA,1);
            end
        elseif q1_NA==3
            for j=1:d_0
                for a=1:d_0
                    for b=1:d_0
                        temp=gamrnd(alpha0_0*lambda0_NA_A,1);
                        temp=temp/sum(temp);
                        lambda_temp(:,j,a,b)=temp;
                    end
                end
            end
            lambda_aa=omega1_1*lambda_x_NA_A+omega0_1*lambda_temp;
            C_0(1:3)=randsample(d_0,3,'true');
            Y_0(1)=normrnd(mu_R(C_0(1)),sigma0_NA,1);
            Y_0(2)=normrnd(mu_R(C_0(2)),sigma0_NA,1);
            Y_0(3)=normrnd(mu_R(C_0(3)),sigma0_NA,1);
            for i=4:T
                C_0(i)=randsample(d_0,1,'true',[lambda_aa(1,C_0(i-1),C_0(i-2),C_0(i-3)),...
                         lambda_aa(2,C_0(i-1),C_0(i-2),C_0(i-3)),lambda_aa(3,C_0(i-1),C_0(i-2),C_0(i-3))]);
                Y_0(i)=normrnd(mu_R(C_0(i)),sigma0_NA,1);
            end
        end
   else
       %%% Group B
       % mixed effect: lambda
       lambda_temp=tensor(zeros(repmat(d_0,1,q2_NA+1)),repmat(d_0,1,q2_NA+1));
       if q2_NA==2
            for j=1:d_0
                for a=1:d_0
                    temp=gamrnd(alpha0_0*lambda0_NA_B,1);
                    temp=temp/sum(temp);
                    lambda_temp(:,j,a)=temp;
                end
            end
            lambda_aa=omega1_2*lambda_x_NA_B+omega0_2*lambda_temp;
            % Generate sequences
            C_0(1:2)=randsample(d_0,2,'true');
            Y_0(1)=normrnd(mu_R(C_0(1)),sigma0_NA,1);
            Y_0(2)=normrnd(mu_R(C_0(2)),sigma0_NA,1);
            for i=3:T
                C_0(i)=randsample(d_0,1,'true',[lambda_aa(1,C_0(i-1),C_0(i-2)),...
                         lambda_aa(2,C_0(i-1),C_0(i-2)),...
                         lambda_aa(3,C_0(i-1),C_0(i-2))]);
                Y_0(i)=normrnd(mu_R(C_0(i)),sigma0_NA,1);
            end
        elseif q2_NA==3
            for j=1:d_0
                for a=1:d_0
                    for b=1:d_0
                        temp=gamrnd(alpha0_0*lambda0_NA_B,1);
                        temp=temp/sum(temp);
                        lambda_temp(:,j,a,b)=temp;
                    end
                end
            end
            lambda_aa=omega1_2*lambda_x_NA_B+omega0_2*lambda_temp;
            % Generate sequences
            C_0(1:3)=randsample(d_0,3,'true');
            Y_0(1)=normrnd(mu_R(C_0(1)),sigma0_NA,1);
            Y_0(2)=normrnd(mu_R(C_0(2)),sigma0_NA,1);
            Y_0(3)=normrnd(mu_R(C_0(3)),sigma0_NA,1);
            for i=4:T
                C_0(i)=randsample(d_0,1,'true',[lambda_aa(1,C_0(i-1),C_0(i-2),C_0(i-3)),...
                         lambda_aa(2,C_0(i-1),C_0(i-2),C_0(i-3)),...
                         lambda_aa(3,C_0(i-1),C_0(i-2),C_0(i-3))]);
                Y_0(i)=normrnd(mu_R(C_0(i)),sigma0_NA,1);
            end
        elseif q2_NA==4
            for j=1:d_0
                for a=1:d_0
                    for b=1:d_0
                        for c=1:d_0
                            temp=gamrnd(alpha0_0*lambda0_NA_B,1);
                            temp=temp/sum(temp);
                            lambda_temp(:,j,a,b,c)=temp;
                        end
                    end
                end
            end
            lambda_aa=omega1_2*lambda_x_NA_B+omega0_2*lambda_temp;
            % Generate sequences
            C_0(1:4)=randsample(d_0,4,'true');
            Y_0(1)=normrnd(mu_R(C_0(1)),sigma0_NA,1);
            Y_0(2)=normrnd(mu_R(C_0(2)),sigma0_NA,1);
            Y_0(3)=normrnd(mu_R(C_0(3)),sigma0_NA,1);
            Y_0(4)=normrnd(mu_R(C_0(4)),sigma0_NA,1);
            for i=5:T
                C_0(i)=randsample(d_0,1,'true',[lambda_aa(1,C_0(i-1),C_0(i-2),C_0(i-3),C_0(i-4)),...
                         lambda_aa(2,C_0(i-1),C_0(i-2),C_0(i-3),C_0(i-4)),...
                         lambda_aa(3,C_0(i-1),C_0(i-2),C_0(i-3),C_0(i-4))]);
                Y_0(i)=normrnd(mu_R(C_0(i)),sigma0_NA,1);
            end
        end

   end
   Y_NA_K{aa}=Y_0;
   C_NA_K{aa}=C_0;
%    C_NA_total_true((T*(aa-1)+1):(T*aa))=C_NA_K{aa};
end

% plot generated sequences
if 1==0
    figure;
    for aa=1:KK
        plot(Y_NA_K{aa});hold on;   
    end
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%
%%% MHOHMM learning %%%
%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% Training and testing sets %%%%%%%%%%%%
%%% k-fold cross validation %%%
CV_num=5;

%%%%%% Model N %%%%%%
Train_Num_N=KK*(4/5);
Test_Num_N=KK*(1/5);
N_cv=Train_Num_N/CV_num;
N_A_cv=N_cv/2;
N_B_cv=N_cv/2;

% random idx 
TrainIdx_N_A=randsample(find(KK_level==1),Train_Num_N/2,false);
TrainIdx_N_B=randsample(find(KK_level==2),Train_Num_N/2,false);
TrainIdx_N=[TrainIdx_N_A,TrainIdx_N_B];
Y_Train_N=Y_N_K(TrainIdx_N);
level_Train_N=[ones(1,Train_Num_N/2),repmat(2,1,Train_Num_N/2)];

TestIdx_N_A=setdiff(find(KK_level==1),TrainIdx_N_A);
TestIdx_N_B=setdiff(find(KK_level==2),TrainIdx_N_B);
TestIdx_N=[TestIdx_N_A,TestIdx_N_B];
Y_Test_N=Y_N_K(TestIdx_N);
level_Test_N=[ones(1,Test_Num_N/2),repmat(2,1,Test_Num_N/2)];


%%%%%% Model NA %%%%%%
Train_Num_NA=KK*(4/5);
Test_Num_NA=KK*(1/5);

% random idx 
TrainIdx_NA_A=randsample(find(KK_level==1),Train_Num_NA/2,false);
TrainIdx_NA_B=randsample(find(KK_level==2),Train_Num_NA/2,false);
TrainIdx_NA=[TrainIdx_NA_A,TrainIdx_NA_B];
Y_Train_NA=Y_NA_K(TrainIdx_NA);
level_Train_NA=[ones(1,Train_Num_NA/2),repmat(2,1,Train_Num_NA/2)];

TestIdx_NA_A=setdiff(find(KK_level==1),TrainIdx_NA_A);
TestIdx_NA_B=setdiff(find(KK_level==2),TrainIdx_NA_B);
TestIdx_NA=[TestIdx_NA_A,TestIdx_NA_B];
Y_Test_NA=Y_NA_K(TestIdx_NA);
level_Test_NA=[ones(1,Test_Num_NA/2),repmat(2,1,Test_Num_NA/2)];


%%
%%%%%%%%%% Stage 1 %%%%%%%%%%%%
% In Stage 1, we take out random effects in the hidden process
% to determine the number of states and identify the important lags
% for each Group

%%% Assign Priors 
dmax=d_0;   % for known state space
% dmax=5;	% maximum number of states
qmax=5;	% maximum dependence order
pM=zeros(qmax,dmax);
for j=1:qmax
    pM(j,1:dmax)=exp(-(j*(1:dmax)/2-1)); % prior probability for k_{j}
    pM(j,1:dmax)=pM(j,1:dmax)./sum(pM(j,1:dmax));
end
alpha00=1;     % prior for Dirichlet distribution for lambda00
alpha_x=1;     % prior for Dirichlet distribution for lambda_x
N1=500;     % number of iterations for first stage



%%%%%%%%%% Stage 2 %%%%%%%%%%%%
%%% Assign Priors & initial values
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

    
%% 
%%%%%%%%%%%%%%%%%
%%% k-fold CV %%%
%%%%%%%%%%%%%%%%%
cv_metric=zeros(Str_num,CV_num+1);

for irep = 1:Str_num
    
model_str=model_str_all(irep,:);

for ii=1:CV_num

%% data preparation

Num_cv=N_cv*(CV_num-1);
level_cv=[ones(1,Num_cv/2),repmat(2,1,Num_cv/2)];
level_cv_t=[ones(1,N_A_cv),repmat(2,1,N_B_cv)];
level_cv_t=repmat(level_cv_t,1,2);

% Model N
tempTest=((ii-1)*N_A_cv+1):(ii*N_A_cv);
tempTrain=setdiff(1:(Train_Num_N/2),tempTest);
Train_cv_idx=[TrainIdx_N_A(tempTrain),TrainIdx_N_B(tempTrain)];
Test_cv_idx=[TrainIdx_N_A(tempTest),TrainIdx_N_B(tempTest)];
Y_Train_N_cv=Y_N_K(Train_cv_idx);
Y_Test_N_cv=Y_N_K(Test_cv_idx);

Y_total_N_cv=zeros(Num_cv*T,1);
for aa=1:Num_cv
    Y_total_N_cv((T*(aa-1)+1):(T*aa))=Y_Train_N_cv{aa};
end
T_N_cv=repmat(T,Num_cv,1);

% Model NA
Train_cv_idx=[TrainIdx_NA_A(tempTrain),TrainIdx_NA_B(tempTrain)];
Test_cv_idx=[TrainIdx_NA_A(tempTest),TrainIdx_NA_B(tempTest)];
Y_Train_NA_cv=Y_NA_K(Train_cv_idx);
Y_Test_NA_cv=Y_NA_K(Test_cv_idx);

Y_total_NA_cv=zeros(Num_cv*T,1);
for aa=1:Num_cv
    Y_total_NA_cv((T*(aa-1)+1):(T*aa))=Y_Train_NA_cv{aa};
end
T_NA_cv=repmat(T,Num_cv,1);



% emission distribution
if strcmp(likelihood_type,'Normal') 
    d0init=3;   %1,dmax
    [idx_N, MuY_N, sumd]=kmeans(Y_total_N_cv,d0init);
    tbl=tabulate(idx_N);
    SigmaSqY=sumd./(2.*tbl(:,2));
    MuY_N((d0init+1):dmax)=(min(Y_total_N_cv):(max(Y_total_N_cv)-min(Y_total_N_cv))/...
        (dmax-d0init-1):max(Y_total_N_cv));
    SigmaSqY((d0init+1):dmax)=var(Y_total_N_cv);
    SigmaSqY(SigmaSqY==0)=var(Y_total_N_cv);
    Parameters=cell(1,2);
    Parameters{1}=MuY_N;
    Parameters{2}=SigmaSqY;
    Parameters_N=Parameters;
    % prior
    Parameters_prior=cell(1,8);
    Parameters_prior{1}=mean(Y_total_N_cv); % Mu0
    Parameters_prior{2}=3*var(Y_total_N_cv); % SigmaSq0
    Parameters_prior{3}=1; % kappa0
    Parameters_prior{4}=1; % beta0
    Parameters_prior{5}=0.1*mean(Y_total_N_cv); % Mu_f
    Parameters_prior{6}=0.1*var(Y_total_N_cv); % SigmaSq_f
    Parameters_prior{7}=1; % kappa_r
    Parameters_prior{8}=1; % beta_r
    Parameters_prior_N=Parameters_prior;
end

if strcmp(likelihood_type,'Normal') 
    [idx_NA, MuY_NA, sumd]=kmeans(Y_total_NA_cv,d0init);
    tbl=tabulate(idx_NA);
    SigmaSqY=sumd./(2.*tbl(:,2));
    MuY_NA((d0init+1):dmax)=(min(Y_total_NA_cv):(max(Y_total_NA_cv)-min(Y_total_NA_cv))/...
        (dmax-d0init-1):max(Y_total_NA_cv));
    SigmaSqY((d0init+1):dmax)=var(Y_total_NA_cv);
    SigmaSqY(SigmaSqY==0)=var(Y_total_NA_cv);
    Parameters=cell(1,2);
    Parameters{1}=MuY_NA;
    Parameters{2}=SigmaSqY;
    Parameters_NA=Parameters;
    % prior
    Parameters_prior=cell(1,8);
    Parameters_prior{1}=mean(Y_total_NA_cv); % Mu0
    Parameters_prior{2}=3*var(Y_total_NA_cv); % SigmaSq0
    Parameters_prior{3}=1; % kappa0
    Parameters_prior{4}=1; % beta0
    Parameters_prior{5}=0.1*mean(Y_total_NA_cv); % Mu_f
    Parameters_prior{6}=0.1*var(Y_total_NA_cv); % SigmaSq_f
    Parameters_prior{7}=1; % kappa_r
    Parameters_prior{8}=1; % beta_r
    Parameters_prior_NA=Parameters_prior;
end

%%
%%%%%%%%%%%%%%%
%%% Model N %%%
%%%%%%%%%%%%%%%
beta=0.5; % initial fixed effect
C_total=idx_N;
CC_K=cell(1,Num_cv); % initial state sequences 
for aa=1:Num_cv
    CC_K{aa}=C_total((T*(aa-1)+1):(T*aa));
end
%%% Sampler for Stage 1
disp(['------------------ CV ',num2str(ii),' ------------------'])
fprintf('\n--- MHOHMM (Model N) --- \n\n');
fprintf('\n--- First Stage --- \n\n');
[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Num_cv,level_N,level_cv,CC_K,Y_Train_N_cv,...
    Y_total_N_cv,T_N_cv,alpha_x,alpha00,Parameters_N,...
    beta,likelihood_type,Parameters_prior_N,pM,model_str);

%%% Sampler for Stage 2
fprintf('\n --- Second Stage --- \n\n');
pause(1);
[lambda_x_Mix_N,lambda_x0_Mix_N,alpha_x0_Mix_N,alpha_x1_Mix_N,omega_Mix_N,...
    pi_Mix_N,mu_Mix_N,sigma_Mix_N,beta_Mix_N,Sigma_r_Mix_N,L00_Mix_N,...
    State_select_Mix_N,p00_Mix_N,K00_Mix_N,ind00_Mix_N,likeli_Mix_N]=...
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Num_cv,...
    level_N,level_cv,C_total,CC_K,Xnew,Cnew,Y_Train_N_cv,Y_total_N_cv,...
    T_N_cv,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,....
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_N,model_str);


%%
%%%%%%%%%%%%%%%%
%%% Model NA %%%
%%%%%%%%%%%%%%%%
%%%%%%%%%% Stage 1 %%%%%%%%%%%%
%%% Assign Priors 
% emission distribution

beta=0.5; % initial fixed effect
C_total=idx_NA;
CC_K=cell(1,Num_cv); % initial state sequences 
for aa=1:Num_cv
    CC_K{aa}=C_total((T*(aa-1)+1):(T*aa));
end

%%% Sampler for Stage 1
fprintf('\n--- MHOHMM (Model NA) --- \n\n');
fprintf('\n--- First Stage --- \n\n');
[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Num_cv,level_N,level_cv,CC_K,Y_Train_NA_cv,...
    Y_total_NA_cv,T_NA_cv,alpha_x,alpha00,Parameters_NA,...
    beta,likelihood_type,Parameters_prior_NA,pM,model_str);

%%% Sampler for Stage 2
fprintf('\n --- Second Stage --- \n\n');
pause(1);
[lambda_x_Mix_NA,lambda_x0_Mix_NA,alpha_x0_Mix_NA,alpha_x1_Mix_NA,omega_Mix_NA,pi_Mix_NA,...
    mu_Mix_NA,sigma_Mix_NA,beta_Mix_NA,Sigma_r_Mix_NA,L00_Mix_NA,...
    State_select_Mix_NA,p00_Mix_NA,K00_Mix_NA,ind00_Mix_NA,likeli_Mix_NA]=...
    MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,Num_cv,...
    level_N,level_cv,C_total,CC_K,Xnew,Cnew,Y_Train_NA_cv,Y_total_NA_cv,...
    T_NA_cv,pigamma,alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,....
    alpha_a0,alpha_b0,alpha_a1,alpha_b1,omega,omega_a,omega_b,...
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_NA,model_str);
%% 
if classification==1
    sampsize=1000;
    num_para=3;
    KK_t=length(level_cv_t); % num of testing units
    Y_Test_cv=[Y_Test_N_cv,Y_Test_NA_cv]; % data for testing
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
%     AUC_Accuracy(str,1)=auc_M;
%     AUC_Accuracy(str,2)=sum(class_test_MHOHMM==classTrue)/KK_t;
    if cv_metric_select==1 % AUC
       cv_metric(irep,ii)=auc_M; 
    elseif cv_metric_select==2 % Accuracy
       cv_metric(irep,ii)=sum(class_test_MHOHMM==classTrue)/KK_t; 
    end
    
%     ROC_x(:,str)=x_M;
%     ROC_y(:,str)=y_M;

end


end

end

cv_metric_mean=mean(cv_metric(:,1:(end-1)),2);
cv_metric(:,end)=cv_metric_mean;
[~,str_id]=max(cv_metric_mean);


%%% Cross validation set %%%
Model_Name={'Hf';'Hfr';'Hf.Of';'Hfr.Of';'Hf.Or';'Hfr.Or';...
   'Hf.Ofr';'Hfr.Ofr'};
temp=table(Model_Name);
Vname=cell(1,CV_num+1);
for cc=1:CV_num
    tempcv=['CV',num2str(cc)];
    Vname{cc}=tempcv;
end
Vname{end}='Mean';
cv_metric=array2table(cv_metric,'VariableNames',Vname);
if cv_metric_select==1 % AUC
    filename1=['Result_AOAS/Classification/AUC_CV',num2str(CV_num),'_Seed_',num2str(seed),'.xlsx'];
elseif cv_metric_select==2 % Accuracy
    filename1=['Result_AOAS/Classification/Accuracy_CV',num2str(CV_num),'_Seed_',num2str(seed),'.xlsx'];
end
writetable([temp,cv_metric],filename1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Final Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if str_id==8 % true generative model (Hfr.Ofr)
    
    model_str_best=model_str_all(str_id,:);
    Y_Train_total_N=zeros(Train_Num_N*T,1);
    for aa=1:Train_Num_N
        Y_Train_total_N((T*(aa-1)+1):(T*aa))=Y_Train_N{aa};
    end
    T_N=repmat(T,Train_Num_N,1);
    Y_Train_total_NA=zeros(Train_Num_NA*T,1);
    for aa=1:Train_Num_NA
        Y_Train_total_NA((T*(aa-1)+1):(T*aa))=Y_Train_NA{aa};
    end
    T_NA=repmat(T,Train_Num_NA,1);
    
    % emission distribution
    if strcmp(likelihood_type,'Normal') 
        d0init=3;   %1,dmax
        [idx_N, MuY_N, sumd]=kmeans(Y_Train_total_N,d0init);
        tbl=tabulate(idx_N);
        SigmaSqY=sumd./(2.*tbl(:,2));
        MuY_N((d0init+1):dmax)=(min(Y_Train_total_N):(max(Y_Train_total_N)-min(Y_Train_total_N))/...
            (dmax-d0init-1):max(Y_Train_total_N));
        SigmaSqY((d0init+1):dmax)=var(Y_Train_total_N);
        SigmaSqY(SigmaSqY==0)=var(Y_Train_total_N);
        Parameters=cell(1,2);
        Parameters{1}=MuY_N;
        Parameters{2}=SigmaSqY;
        Parameters_N=Parameters;
        % prior
        Parameters_prior=cell(1,8);
        Parameters_prior{1}=mean(Y_Train_total_N); % Mu0
        Parameters_prior{2}=3*var(Y_Train_total_N); % SigmaSq0
        Parameters_prior{3}=1; % kappa0
        Parameters_prior{4}=1; % beta0
        Parameters_prior{5}=0.1*mean(Y_Train_total_N); % Mu_f
        Parameters_prior{6}=0.1*var(Y_Train_total_N); % SigmaSq_f
        Parameters_prior{7}=1; % kappa_r
        Parameters_prior{8}=1; % beta_r
        Parameters_prior_N=Parameters_prior;
    end
    
    
    if strcmp(likelihood_type,'Normal') 
        [idx_NA, MuY_NA, sumd]=kmeans(Y_Train_total_NA,d0init);
        tbl=tabulate(idx_NA);
        SigmaSqY=sumd./(2.*tbl(:,2));
        MuY_NA((d0init+1):dmax)=(min(Y_Train_total_NA):(max(Y_Train_total_NA)-min(Y_Train_total_NA))/...
            (dmax-d0init-1):max(Y_Train_total_NA));
        SigmaSqY((d0init+1):dmax)=var(Y_Train_total_NA);
        SigmaSqY(SigmaSqY==0)=var(Y_Train_total_NA);
        Parameters=cell(1,2);
        Parameters{1}=MuY_NA;
        Parameters{2}=SigmaSqY;
        Parameters_NA=Parameters;
        % prior
        Parameters_prior=cell(1,8);
        Parameters_prior{1}=mean(Y_Train_total_NA); % Mu0
        Parameters_prior{2}=3*var(Y_Train_total_NA); % SigmaSq0
        Parameters_prior{3}=1; % kappa0
        Parameters_prior{4}=1; % beta0
        Parameters_prior{5}=0.1*mean(Y_Train_total_NA); % Mu_f
        Parameters_prior{6}=0.1*var(Y_Train_total_NA); % SigmaSq_f
        Parameters_prior{7}=1; % kappa_r
        Parameters_prior{8}=1; % beta_r
        Parameters_prior_NA=Parameters_prior;
    end

    
    %%% save data for replications of final models %%%
    filename0=['Result_AOAS/Classification/Final_data_Seed_',num2str(seed),'.mat'];
    save(filename0,'model_str_best','idx_N','Train_Num_N','T','dmax','qmax','N1',...
        'level_N','level_Train_N','Y_Train_N','Y_Train_total_N','T_N','alpha_x',...
        'alpha00','Parameters_N','likelihood_type','Parameters_prior_N','pM',...
        'simsize','burnin','gap','pigamma','alpha_x0','alpha_x1','alpha00',...
        'alpha_a0','alpha_b0','alpha_a1','alpha_b1','omega','omega_a','omega_b',...
        'idx_NA','Train_Num_NA','level_Train_NA','Y_Train_NA','Y_Train_total_NA',...
        'T_NA','Parameters_NA','Parameters_prior_NA','level_Test_N',...
        'level_Test_NA','Y_Test_N','Y_Test_NA','Test_Num_N','Test_Num_NA')  
    

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




%% true generative model (Hfr.Ofr)
if classification==1
    sampsize=1000;
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
end



%%
%%%%%%%%%%%%%%%%%%%%%%
%%% HOHMM learning %%%
%%%%%%%%%%%%%%%%%%%%%%
model_str_HOHMM=[0,0,0,0,1];
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
alpha0=1;

[Mact_L, C_total, CC_K, Xnew, Cnew, Parameters, mu_r_K, beta, ...
    lambda_x, lambda_x0, lambda00]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,...
    Train_Num_N,level_N,level_Train_N,CC_K,Y_Train_N,...
    Y_Train_total_N,T_N,alpha_x,alpha00,...
    Parameters_N,beta,likelihood_type,Parameters_prior_N,pM,model_str_HOHMM);


% [Mact, C_total, CC_K, Xnew, Cnew, Parameters, lambda0, ~]=HOHMM_str_stage1(...
%     dmax,qmax,N1,Train_Num_N,CC_K,Y_Train_N,Y_Train_total_N,T_N,alpha0,...
%     Parameters_N,likelihood_type,Parameters_prior_H,pM);


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


% [lambda_storage_H_N, lambda0_storage_H_N, pi_storage_H_N, mu_storage_H_N, sigma_storage_H_N,...
%     L00_H_N, State_select_H_N, p00_H_N, K00_H_N, ind00_H_N,likeli_H_N]=...
%     HOHMM_str_stage2(dmax,qmax,N1,Mact,simsize,burnin,gap,Train_Num_N,...
%     C_total,CC_K,Xnew,Cnew,Y_Train_N,Y_Train_total_N,T_N,pigamma,alpha0,...
%     lambda0,Parameters,likelihood_type,Parameters_prior_H);


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

% [Mact, C_total, CC_K, Xnew, Cnew, Parameters, lambda0, ~]=HOHMM_str_stage1(...
%     dmax,qmax,N1,Train_Num_NA,CC_K,Y_Train_NA,Y_Train_total_NA,T_NA,alpha0,...
%     Parameters_NA,likelihood_type,Parameters_prior_H,pM);


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
 
% [lambda_storage_H_NA, lambda0_storage_H_NA, pi_storage_H_NA, mu_storage_H_NA, sigma_storage_H_NA,...
%     L00_H_NA, State_select_H_NA, p00_H_NA, K00_H_NA, ind00_H_NA,likeli_H_NA]=...
%     HOHMM_str_stage2(dmax,qmax,N1,Mact,simsize,burnin,gap,Train_Num_NA,...
%     C_total,CC_K,Xnew,Cnew,Y_Train_NA,Y_Train_total_NA,T_NA,pigamma,alpha0,...
%     lambda0,Parameters,likelihood_type,Parameters_prior_H);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Classification HOHMM %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if classification==1
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
end


%%
%%%%%%%%%%%%%%%%%%%%%
%%% MHMM learning %%%
%%%%%%%%%%%%%%%%%%%%%%
model_str_MHMM=model_str_best;
model_str_MHMM(5)=0;

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

if classification==1
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
end


%%
%%%%%%%%%%%%%%%%%%%
%%%%% Results %%%%%
%%%%%%%%%%%%%%%%%%%

%%% Testing set %%%
Model={Model_Name{str_id};'HOHMM';'MHMM'};
temp=table(Model);
results=array2table(results_final,'VariableNames',{'AUC','Accuracy','Sensivity','Specificity','Precision','F1-score'});
filename2=['Result_AOAS/Classification/Final_classification_test_Seed_',num2str(seed),'.xlsx'];
writetable([temp,results],filename2);

end

toc
compute_time=toc;

end

