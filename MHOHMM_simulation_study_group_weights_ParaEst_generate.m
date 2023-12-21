%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --              Mixed Higher Order Hidden Markov Model              -- %%%
%%% --                       Simulation study                           -- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% -- By Ying Liao, Last modified in December, 2020 -- %

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


% dbstop if error;
%%
%%%%%%%%%%%%%%%%
%%% Set seed %%%
%%%%%%%%%%%%%%%%
NRep = 16; % Number of replications
seed = (1 : NRep) + 0; % Set random seeds

compute_time=zeros(1,NRep);

%% Parallel implementation
parfor irep = 1:NRep
% for irep = 1:NRep  
    temp=HMMExp(seed(irep));
    compute_time(irep)=temp;
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

% model 1: 'Normal' 
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

       
% model 2: 'Abnormal' 
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
%%% save parameters for figures %%%
lambda_x0_figure=cell(2,Str_num); % class X model structures
mu_figure=cell(2,Str_num); 
sigma_figure=cell(2,Str_num);
beta_figure=cell(2,Str_num);
Sigma_r_figure=cell(2,Str_num);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%
%%% MHOHMM learning %%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Training and testing sets %%%%%%%%%%%%
%%%%%% Model N %%%%%%
Train_Num_N=KK*(4/5);
Test_Num_N=KK*(1/5);
% training idx for each group
TrainIdx_N_A=randsample(find(KK_level==1),Train_Num_N/2,false);
TrainIdx_N_A=sort(TrainIdx_N_A);
TrainIdx_N_B=randsample(find(KK_level==2),Train_Num_N/2,false);
TrainIdx_N_B=sort(TrainIdx_N_B);
% testing idx for each group
TestIdx_N_A=setdiff(find(KK_level==1),TrainIdx_N_A);
TestIdx_N_B=setdiff(find(KK_level==2),TrainIdx_N_B);
% training idx (all)
TrainIdx_N=[TrainIdx_N_A,TrainIdx_N_B];
% testing idx (all)
TestIdx_N=[TestIdx_N_A,TestIdx_N_B];
% training sequences
Y_Train_N=Y_N_K(TrainIdx_N);
% testing sequences
Y_Test_N=Y_N_K(TestIdx_N);
% group label for training set
level_Train_N=[ones(1,Train_Num_N/2),repmat(2,1,Train_Num_N/2)];
% group label for testing set
level_Test_N=[ones(1,Test_Num_N/2),repmat(2,1,Test_Num_N/2)];
% training sequences (all)
Y_Train_total_N=zeros(Train_Num_N*T,1);
for aa=1:Train_Num_N
    Y_Train_total_N((T*(aa-1)+1):(T*aa))=Y_Train_N{aa};
end
T_N=repmat(T,Train_Num_N,1);

%%%%%% Model NA %%%%%%
Train_Num_NA=KK*(4/5);
Test_Num_NA=KK*(1/5);
% training idx for each group
TrainIdx_NA_A=randsample(find(KK_level==1),Train_Num_NA/2,false);
TrainIdx_NA_A=sort(TrainIdx_NA_A);
TrainIdx_NA_B=randsample(find(KK_level==2),Train_Num_NA/2,false);
TrainIdx_NA_B=sort(TrainIdx_NA_B);
% testing idx for each group
TestIdx_NA_A=setdiff(find(KK_level==1),TrainIdx_NA_A);
TestIdx_NA_B=setdiff(find(KK_level==2),TrainIdx_NA_B);
% training idx (all)
TrainIdx_NA=[TrainIdx_NA_A,TrainIdx_NA_B];
% testing idx (all)
TestIdx_NA=[TestIdx_NA_A,TestIdx_NA_B];
% training sequences
Y_Train_NA=Y_NA_K(TrainIdx_NA);
% testing sequences
Y_Test_NA=Y_NA_K(TestIdx_NA);
% group label for training set
level_Train_NA=[ones(1,Train_Num_NA/2),repmat(2,1,Train_Num_NA/2)];
% group label for testing set
level_Test_NA=[ones(1,Test_Num_NA/2),repmat(2,1,Test_Num_NA/2)];
% training sequences (all)
Y_Train_total_NA=zeros(Train_Num_NA*T,1);
for aa=1:Train_Num_NA
    Y_Train_total_NA((T*(aa-1)+1):(T*aa))=Y_Train_NA{aa};
end
T_NA=repmat(T,Train_Num_NA,1);

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


%% Iteration
ObsDimension=size(Y_Train_total_N,2);
para_est_str=nan(2*((5+dmax)*level_N+(2*dmax+2)*ObsDimension+2),2*Str_num);

for str=1:Str_num
model_str=model_str_all(str,:);

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
    Parameters_N,beta,likelihood_type,Parameters_prior_N,pM,model_str);


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
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_N,model_str);

%%% parameter for figures
lambda_x0_figure{1,str}=lambda_x0_Mix_N;
mu_figure{1,str}=mu_Mix_N;
sigma_figure{1,str}=sigma_Mix_N;
beta_figure{1,str}=beta_Mix_N;
Sigma_r_figure{1,str}=Sigma_r_Mix_N;

%%%% parameter estimation %%%%
Mact=Mact_L{1};
MactFirstStage=Mact;
MProp=(MactFirstStage>1);
MProp=mean(MProp(floor(N1/2):N1,:),1);
ind00=find(MProp>0.5);
if sum(ind00)==0
   ind00=1;
end
para_est_str(1,str)=max(ind00); % order q_1
para_est_str(2,str)=MProp(max(ind00)); % posterior probability

Mact=Mact_L{2};
MactFirstStage=Mact;
MProp=(MactFirstStage>1);
MProp=mean(MProp(floor(N1/2):N1,:),1);
ind00=find(MProp>0.5);
if sum(ind00)==0
   ind00=1;
end
para_est_str(3,str)=max(ind00); % order q_2
para_est_str(4,str)=MProp(max(ind00)); % posterior probability

if model_str(2)==1 % random effect in hidden process
    temp=mean(omega_Mix_N,3);
    temp=temp';
    temp2=std(omega_Mix_N,0,3);
    temp2=temp2';
    
    para_est_str(5:6,str)=temp(:,1); % weights for group 1
    para_est_str(7:8,str)=temp2(:,1); % std
    para_est_str(9:10,str)=temp(:,2); % weights for group 2
    para_est_str(11:12,str)=temp2(:,2); % std
    
    temp3=mean(alpha_x0_Mix_N,2);
    temp4=std(alpha_x0_Mix_N,0,2);
    
    para_est_str(13,str)=temp3(1); % alpha_x0 for group 1
    para_est_str(14,str)=temp4(1); % std
    para_est_str(17,str)=temp3(2); % alpha_x0 for group 2 
    para_est_str(18,str)=temp4(2); % std
end
temp5=mean(alpha_x1_Mix_N,2);
temp6=std(alpha_x1_Mix_N,0,2);
para_est_str(15,str)=temp5(1); % alpha_x1 for group 1
para_est_str(16,str)=temp6(1); % std
para_est_str(19,str)=temp5(2); % alpha_x1 for group 2 
para_est_str(20,str)=temp6(2); % std

temp7=mean(lambda_x0_Mix_N,3);
temp8=std(lambda_x0_Mix_N,0,3);
para_est_str(21:23,str)=temp7(:,1); % state preference for group 1
para_est_str(24:26,str)=temp8(:,1); % std
para_est_str(27:29,str)=temp7(:,2); % state preference for group 2
para_est_str(30:32,str)=temp8(:,2); % std

para_est_str(33:35,str)=mean(mu_Mix_N,2);
para_est_str(36:38,str)=std(mu_Mix_N,0,2); % std

para_est_str(39:41,str)=mean(sqrt(sigma_Mix_N),2);
para_est_str(42:44,str)=std(sqrt(sigma_Mix_N),0,2); % std

if model_str(3)==1 % fixed effect in obs process
    para_est_str(45,str)=mean(beta_Mix_N); % beta
    para_est_str(46,str)=std(beta_Mix_N);
end
if model_str(4)==1 % random effect in obs process
    para_est_str(47,str)=mean(sqrt(Sigma_r_Mix_N)); 
    para_est_str(48,str)=std(sqrt(Sigma_r_Mix_N)); 
end
para_est_str(49,str)=mode(L00_Mix_N); % number of states
para_est_str(50,str)=sum(L00_Mix_N==mode(L00_Mix_N))/length(L00_Mix_N);
para_est_str(51,str)=mean(likeli_Mix_N); % likelihood
para_est_str(52,str)=std(likeli_Mix_N);
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
    likelihood_type,Parameters_prior_NA,pM,model_str);


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
    Parameters,likelihood_type,mu_r_K,beta,Parameters_prior_NA,model_str);

%%% parameter for figures
lambda_x0_figure{2,str}=lambda_x0_Mix_NA;
mu_figure{2,str}=mu_Mix_NA;
sigma_figure{2,str}=sigma_Mix_NA;
beta_figure{2,str}=beta_Mix_NA;
Sigma_r_figure{2,str}=Sigma_r_Mix_NA;


%%%% parameter estimation %%%%
Mact=Mact_L{1};
MactFirstStage=Mact;
MProp=(MactFirstStage>1);
MProp=mean(MProp(floor(N1/2):N1,:),1);
ind00=find(MProp>0.5);
if sum(ind00)==0
   ind00=1;
end
para_est_str(1,str+Str_num)=max(ind00); % order q_1
para_est_str(2,str+Str_num)=MProp(max(ind00)); % posterior probability

Mact=Mact_L{2};
MactFirstStage=Mact;
MProp=(MactFirstStage>1);
MProp=mean(MProp(floor(N1/2):N1,:),1);
ind00=find(MProp>0.5);
if sum(ind00)==0
   ind00=1;
end
para_est_str(3,str+Str_num)=max(ind00); % order q_2
para_est_str(4,str+Str_num)=MProp(max(ind00)); % posterior probability

if model_str(2)==1 % random effect in hidden process
    temp=mean(omega_Mix_NA,3); % level X (w_0,w_1)
    temp=temp';
    temp2=std(omega_Mix_NA,0,3);
    temp2=temp2';
    
    para_est_str(5:6,str+Str_num)=temp(:,1); % weights for group 1
    para_est_str(7:8,str+Str_num)=temp2(:,1); % std
    para_est_str(9:10,str+Str_num)=temp(:,2); % weights for group 2
    para_est_str(11:12,str+Str_num)=temp2(:,2); % std
    
    temp3=mean(alpha_x0_Mix_NA,2);
    temp4=std(alpha_x0_Mix_NA,0,2);
    
    para_est_str(13,str+Str_num)=temp3(1); % alpha_x0 for group 1
    para_est_str(14,str+Str_num)=temp4(1); % std
    para_est_str(17,str+Str_num)=temp3(2); % alpha_x0 for group 2 
    para_est_str(18,str+Str_num)=temp4(2); % std
end
temp5=mean(alpha_x1_Mix_NA,2);
temp6=std(alpha_x1_Mix_NA,0,2);
para_est_str(15,str+Str_num)=temp5(1); % alpha_x1 for group 1
para_est_str(16,str+Str_num)=temp6(1); % std
para_est_str(19,str+Str_num)=temp5(2); % alpha_x1 for group 2 
para_est_str(20,str+Str_num)=temp6(2); % std

temp7=mean(lambda_x0_Mix_NA,3);
temp8=std(lambda_x0_Mix_NA,0,3);
para_est_str(21:23,str+Str_num)=temp7(:,1); % state preference for group 1
para_est_str(24:26,str+Str_num)=temp8(:,1); % std
para_est_str(27:29,str+Str_num)=temp7(:,2); % state preference for group 2
para_est_str(30:32,str+Str_num)=temp8(:,2); % std

para_est_str(33:35,str+Str_num)=mean(mu_Mix_NA,2);
para_est_str(36:38,str+Str_num)=std(mu_Mix_NA,0,2); % std

para_est_str(39:41,str+Str_num)=mean(sqrt(sigma_Mix_NA),2);
para_est_str(42:44,str+Str_num)=std(sqrt(sigma_Mix_NA),0,2); % std

if model_str(3)==1 % fixed effect in obs process
    para_est_str(45,str+Str_num)=mean(beta_Mix_NA); % beta
    para_est_str(46,str+Str_num)=std(beta_Mix_NA);
end
if model_str(4)==1 % random effect in obs process
    para_est_str(47,str+Str_num)=mean(sqrt(Sigma_r_Mix_NA)); 
    para_est_str(48,str+Str_num)=std(sqrt(Sigma_r_Mix_NA)); 
end
para_est_str(49,str+Str_num)=mode(L00_Mix_NA); % number of states
para_est_str(50,str+Str_num)=sum(L00_Mix_NA==mode(L00_Mix_NA))/length(L00_Mix_NA);
para_est_str(51,str+Str_num)=mean(likeli_Mix_NA); % likelihood
para_est_str(52,str+Str_num)=std(likeli_Mix_NA);



end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Parameter estimation %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Parameter={'q1';'q1_prob';'q2';'q2_prob';'w0_1';'w1_1';'w0_1_std';'w1_1_std';...
    'w0_2';'w1_2';'w0_2_std';'w1_2_std';'alpha0_1';'alpha0_1_std';'alpha1_1';'alpha1_1_std';...
    'alpha0_2';'alpha0_2_std';'alpha1_2';'alpha1_2_std';'lambda1_1';'lambda1_2';'lambda1_3';...
    'lambda1_1_std';'lambda1_2_std';'lambda1_3_std';'lambda2_1';'lambda2_2';'lambda2_3';...
    'lambda2_1_std';'lambda2_2_std';'lambda2_3_std';'mu1';'mu2';'mu3';...
    'mu1_std';'mu2_std';'mu3_std';'sigma1';'sigma2';'sigma3';'sigma1_std';'sigma2_std';'sigma3_std';...
    'beta';'beta_std';'sigma_r';'sigma_r_std';'StateNum';'StateNum_prob';'Loglikeli';'Loglikeli_std'};
temp2=table(Parameter);

para_Mix=para_est_str;

para_true=nan(2*((5+dmax)*level_N+(2*dmax+2)*ObsDimension+2),2);
% model 1
para_true(1,1)=q1_A;
para_true(3,1)=q2_A;
para_true(5,1)=1-omega1_1;
para_true(6,1)=omega1_1;
para_true(9,1)=1-omega1_2;
para_true(10,1)=omega1_2;
para_true(13,1)=alpha0_0;
para_true(17,1)=alpha0_0;
para_true(15,1)=alpha1_0;
para_true(19,1)=alpha1_0;
para_true(21:23,1)=lambda0_N_A;
para_true(27:29,1)=lambda0_N_B;
para_true(33:35,1)=mu0_N';
para_true(39:41,1)=[sigma0_N;sigma0_N;sigma0_N];
para_true(45,1)=beta_N0;
para_true(47,1)=sigma0_R;
para_true(49,1)=d_0;
% model 2
para_true(1,2)=q1_NA;
para_true(3,2)=q2_NA;
para_true(5,2)=1-omega1_1;
para_true(6,2)=omega1_1;
para_true(9,2)=1-omega1_2;
para_true(10,2)=omega1_2;
para_true(13,2)=alpha0_0;
para_true(17,2)=alpha0_0;
para_true(15,2)=alpha1_0;
para_true(19,2)=alpha1_0;
para_true(21:23,2)=lambda0_NA_A;
para_true(27:29,2)=lambda0_NA_B;
para_true(33:35,2)=mu0_NA';
para_true(39:41,2)=[sigma0_NA;sigma0_NA;sigma0_NA];
para_true(45,2)=beta_NA0;
para_true(47,2)=sigma0_R;
para_true(49,2)=d_0;

file_name2=['Result_AOAS/Parameter_estimation/ParaEst_Model_1',...
    '_seed_',num2str(seed),'.xlsx'];
para_temp=[para_true(:,1),para_Mix(:,1:8)];
para=array2table(para_temp,'VariableNames',{'True','Hf','Hfr','Hf_Of','Hfr_Of','Hf_Or','Hfr_Or',...
'Hf_Ofr','Hfr_Ofr'});
writetable([temp2,para],file_name2);

file_name3=['Result_AOAS/Parameter_estimation/ParaEst_Model_2',...
    '_seed_',num2str(seed),'.xlsx'];
para_temp=[para_true(:,2),para_Mix(:,9:16)];
para=array2table(para_temp,'VariableNames',{'True','Hf','Hfr','Hf_Of','Hfr_Of','Hf_Or','Hfr_Or',...
'Hf_Ofr','Hfr_Ofr'});
writetable([temp2,para],file_name3);




%%
%%% save parameters for future use
filename=['Result_AOAS/Parameter_estimation/ParaEst_ForFigure_Seed_',num2str(seed),'.mat'];
save(filename,'lambda_x0_figure','mu_figure','sigma_figure','beta_figure','Sigma_r_figure',...
    'mu0_N','mu0_NA','beta_N0','beta_NA0','sigma0_R','sigma0_N','sigma0_NA',...
    'lambda0_NA_A','lambda0_NA_B','lambda0_N_A','lambda0_N_B')
toc

compute_time=toc;

end







