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

level_N = 2; % number of groups: 2,3,4,5,6      8,10 (sensitivity analysis)
qmax = 9; % 5,7,9 (sensitivity analysis)
dmax = 5; % 3,5,7 (sensitivity analysis)
% folder = ['Result_AOAS/Classification/Revision/Groups_',num2str(level_N),'/'];
folder = ['Result_AOAS/Classification/Revision/Groups_',num2str(level_N),'/All_q_d/q',num2str(qmax),'_d',num2str(dmax),'/'];

% dbstop if error;

%% Parallel implementation
parfor ss = 1:NRep
% for ss = 1:NRep  
    HMMExp(seed(ss),level_N,dmax,qmax,folder);
end


%%
function [] = HMMExp(seed,level_N,dmax,qmax,folder)


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
% Str_num=size(model_str_all,1);
                   
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
omega1_all = repmat([0.6,0.8],1,5);


% alpha
alpha0_0=1; % for random effect
alpha1_0=1; % for fixed effect

% order
% model 1: 'Normal' 
q_N_all =  repmat([2,3],1,5);

% model 2: 'Abnormal'
q_NA_all =  repmat([2,4],1,5);


%% results collection
results_final=nan(3,7); 
% AUC & Accuracy & Sensivity & Specificity & Precision & F1-score & computation time; 
% Best_MHOHMM, HOHMM, MHMM
% cv_metric_select=2; % 1: AUC; 2: Accuracy
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
KK_each = floor(KK/level_N);
KK_level = [];
for kk = 1:(level_N-1)
    KK_level = [KK_level,kk*ones(1,KK_each)];
end
KK_level = [KK_level,level_N*ones(1,KK-KK_each*(level_N-1))];
Num_L=zeros(1,level_N);
for ll=1:level_N
    Num_L(ll)=sum(KK_level==ll);
end
disp('----- Number of sequences in each group -----')
disp(Num_L)

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
    omega1_all = ones(1,level_N);
end 
omega0_all = 1 - omega1_all;

% model 1: 'Normal' Order--2,3,2,3,2,3,2,3,2,3
% state preference
lambda0_N_all = zeros(d_0,level_N);
for kk = 1 : level_N
    lambda0_N_all(1,kk)=betarnd(1,1,1);
    lambda0_N_all(2,kk)=betarnd(1,1,1)*(1-lambda0_N_all(1,kk));
    lambda0_N_all(3,kk)=1-lambda0_N_all(1,kk)-lambda0_N_all(2,kk);
    while(sum(lambda0_N_all(:,kk)<0.1)>0)
        lambda0_N_all(1,kk)=betarnd(1,1,1);
        lambda0_N_all(2,kk)=betarnd(1,1,1)*(1-lambda0_N_all(1,kk));
        lambda0_N_all(3,kk)=1-lambda0_N_all(1,kk)-lambda0_N_all(2,kk);
    end
end

% group-level transition probability
lambda_x_N_all = cell(1,level_N);
for kk = 1 : level_N
    lambda_x_N_temp=tensor(zeros(repmat(d_0,1,q_N_all(kk)+1)),repmat(d_0,1,q_N_all(kk)+1));
    if q_N_all(kk)==2
        for j=1:d_0
            for a=1:d_0
                temp=gamrnd(alpha1_0*lambda0_N_all(:,kk),1);
                temp=temp/sum(temp);
                lambda_x_N_temp(:,j,a)=temp;
            end
        end
    elseif q_N_all(kk)==3
        for j=1:d_0
            for a=1:d_0
                for b=1:d_0
                    temp=gamrnd(alpha1_0*lambda0_N_all(:,kk),1);
                    temp=temp/sum(temp);
                    lambda_x_N_temp(:,j,a,b)=temp;
                end
            end
        end
    elseif q_N_all(kk)==4
        for j=1:d_0
            for a=1:d_0
                for b=1:d_0
                    for c=1:d_0
                        temp=gamrnd(alpha1_0*lambda0_N_all(:,kk),1);
                        temp=temp/sum(temp);
                        lambda_x_N_temp(:,j,a,b,c)=temp;
                    end
                end
            end
        end
    end
    lambda_x_N_all{kk} = lambda_x_N_temp;
end

       
% model 2: 'Abnormal' Order--2,4,2,4,2,4,2,4,2,4
% state preference
lambda0_NA_all = zeros(d_0,level_N);
for kk = 1 : level_N
    lambda0_NA_all(1,kk)=betarnd(1,1,1);
    lambda0_NA_all(2,kk)=betarnd(1,1,1)*(1-lambda0_NA_all(1,kk));
    lambda0_NA_all(3,kk)=1-lambda0_NA_all(1,kk)-lambda0_NA_all(2,kk);
    while(sum(lambda0_NA_all(:,kk)<0.1)>0)
        lambda0_NA_all(1,kk)=betarnd(1,1,1);
        lambda0_NA_all(2,kk)=betarnd(1,1,1)*(1-lambda0_NA_all(1,kk));
        lambda0_NA_all(3,kk)=1-lambda0_NA_all(1,kk)-lambda0_NA_all(2,kk);
    end
end

% group-level transition probability
lambda_x_NA_all = cell(1,level_N);
for kk = 1 : level_N
    lambda_x_NA_temp=tensor(zeros(repmat(d_0,1,q_NA_all(kk)+1)),repmat(d_0,1,q_NA_all(kk)+1));
    if q_NA_all(kk)==2
        for j=1:d_0
            for a=1:d_0
                temp=gamrnd(alpha1_0*lambda0_NA_all(:,kk),1);
                temp=temp/sum(temp);
                lambda_x_NA_temp(:,j,a)=temp;
            end
        end
    elseif q_NA_all(kk)==3
        for j=1:d_0
            for a=1:d_0
                for b=1:d_0
                    temp=gamrnd(alpha1_0*lambda0_NA_all(:,kk),1);
                    temp=temp/sum(temp);
                    lambda_x_NA_temp(:,j,a,b)=temp;
                end
            end
        end
    elseif q_NA_all(kk)==4
        for j=1:d_0
            for a=1:d_0
                for b=1:d_0
                    for c=1:d_0
                        temp=gamrnd(alpha1_0*lambda0_NA_all(:,kk),1);
                        temp=temp/sum(temp);
                        lambda_x_NA_temp(:,j,a,b,c)=temp;
                    end
                end
            end
        end
    end
    lambda_x_NA_all{kk} = lambda_x_NA_temp;
end



if 1==1
    disp('----- State prevalence prob. -----')
    disp('--> Model N: ')
    disp(lambda0_N_all);
    disp('--> Model NA: ')
    disp(lambda0_NA_all);
end
%%
%%%%%%%%%%%%%%%
%%% Model N %%%
%%%%%%%%%%%%%%%
Y_N_K=cell(1,KK); % observation sequences
C_N_K=cell(1,KK); % state sequences

% starting state(s)
for aa=1:KK
   kk_aa=KK_level(aa);
   C_0=zeros(T,1); % state sequence
   Y_0=zeros(T,1); % observation sequence
   % unit-specific mu
   mu_aa=normrnd(0,sigma0_R,1); % mu_a
   mu_R=mu0_N+mu_aa+beta_N0*kk_aa;
   
   % mixed effect: lambda
   lambda_temp=tensor(zeros(repmat(d_0,1,q_N_all(kk_aa)+1)),repmat(d_0,1,q_N_all(kk_aa)+1)); 
   if q_N_all(kk_aa)==2
        for j=1:d_0
            for a=1:d_0
                temp=gamrnd(alpha0_0*lambda0_N_all(:,kk_aa),1);
                temp=temp/sum(temp);
                lambda_temp(:,j,a)=temp;
            end
        end
        lambda_aa=omega1_all(kk_aa)*lambda_x_N_all{kk_aa}+omega0_all(kk_aa)*lambda_temp;
        % Generate sequences
        C_0(1:2)=randsample(d_0,2,'true');
        Y_0(1)=normrnd(mu_R(C_0(1)),sigma0_N,1);
        Y_0(2)=normrnd(mu_R(C_0(2)),sigma0_N,1);
        for i=3:T
            C_0(i)=randsample(d_0,1,'true',[lambda_aa(1,C_0(i-1),C_0(i-2)),...
                     lambda_aa(2,C_0(i-1),C_0(i-2)),lambda_aa(3,C_0(i-1),C_0(i-2))]);
            Y_0(i)=normrnd(mu_R(C_0(i)),sigma0_N,1);
        end
    elseif q_N_all(kk_aa)==3
        for j=1:d_0
            for a=1:d_0
                for b=1:d_0
                    temp=gamrnd(alpha0_0*lambda0_N_all(:,kk_aa),1);
                    temp=temp/sum(temp);
                    lambda_temp(:,j,a,b)=temp;
                end
            end
        end
        lambda_aa=omega1_all(kk_aa)*lambda_x_N_all{kk_aa}+omega0_all(kk_aa)*lambda_temp;
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
    elseif q_N_all(kk_aa)==4
        for j=1:d_0
            for a=1:d_0
                for b=1:d_0
                    for c=1:d_0
                        temp=gamrnd(alpha0_0*lambda0_N_all(:,kk),1);
                        temp=temp/sum(temp);
                        lambda_temp(:,j,a,b,c)=temp;
                    end
                end
            end
        end
        lambda_aa=omega1_all(kk_aa)*lambda_x_N_all{kk_aa}+omega0_all(kk_aa)*lambda_temp;
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

% starting state(s)
for aa=1:KK
    kk_aa=KK_level(aa);
   C_0=zeros(T,1); % state sequence
   Y_0=zeros(T,1); % observation sequence
   % unit-specific mu
   mu_aa=normrnd(0,sigma0_R,1); % mu_a
   mu_R=mu0_NA+mu_aa+beta_NA0*kk_aa;
   
   % mixed effect: lambda
   lambda_temp=tensor(zeros(repmat(d_0,1,q_NA_all(kk_aa)+1)),repmat(d_0,1,q_NA_all(kk_aa)+1));
   if q_NA_all(kk_aa)==2
        for j=1:d_0
            for a=1:d_0
                temp=gamrnd(alpha0_0*lambda0_NA_all(:,kk_aa),1);
                temp=temp/sum(temp);
                lambda_temp(:,j,a)=temp;
            end
        end
        lambda_aa=omega1_all(kk_aa)*lambda_x_NA_all{kk_aa}+omega0_all(kk_aa)*lambda_temp;
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
    elseif q_NA_all(kk_aa)==3
        for j=1:d_0
            for a=1:d_0
                for b=1:d_0
                    temp=gamrnd(alpha0_0*lambda0_NA_all(:,kk_aa),1);
                    temp=temp/sum(temp);
                    lambda_temp(:,j,a,b)=temp;
                end
            end
        end
        lambda_aa=omega1_all(kk_aa)*lambda_x_NA_all{kk_aa}+omega0_all(kk_aa)*lambda_temp;
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
    elseif q_NA_all(kk_aa)==4
        for j=1:d_0
            for a=1:d_0
                for b=1:d_0
                    for c=1:d_0
                        temp=gamrnd(alpha0_0*lambda0_NA_all(:,kk_aa),1);
                        temp=temp/sum(temp);
                        lambda_temp(:,j,a,b,c)=temp;
                    end
                end
            end
        end
        lambda_aa=omega1_all(kk_aa)*lambda_x_NA_all{kk_aa}+omega0_all(kk_aa)*lambda_temp;
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

%%%%%% Model N %%%%%%
Train_Num_N=KK*(4/5);
Test_Num_N=KK*(1/5);

% random idx 
Train_Num_each = floor(Train_Num_N/level_N);
TrainIdx_N = [];
for kk = 1:(level_N-1)
    temp = randsample(find(KK_level==kk),Train_Num_each,false);
    TrainIdx_N = [TrainIdx_N,temp];   
end
temp = randsample(find(KK_level==level_N),Train_Num_each+(Train_Num_N-Train_Num_each*level_N),false);
TrainIdx_N = [TrainIdx_N,temp];
Y_Train_N=Y_N_K(TrainIdx_N);
level_Train_N = KK_level(TrainIdx_N);

TestIdx_N=setdiff(1:KK,TrainIdx_N);
Y_Test_N=Y_N_K(TestIdx_N);
level_Test_N=KK_level(TestIdx_N);


%%%%%% Model NA %%%%%%
Train_Num_NA=KK*(4/5);
Test_Num_NA=KK*(1/5);

% random idx 
Train_Num_each = floor(Train_Num_NA/level_N);
TrainIdx_NA = [];
for kk = 1:(level_N-1)
    temp = randsample(find(KK_level==kk),Train_Num_each,false);
    TrainIdx_NA = [TrainIdx_NA,temp];   
end
temp = randsample(find(KK_level==level_N),Train_Num_each+(Train_Num_NA-Train_Num_each*level_N),false);
TrainIdx_NA = [TrainIdx_NA,temp];
Y_Train_NA=Y_NA_K(TrainIdx_NA);
level_Train_NA=KK_level(TrainIdx_NA);

TestIdx_NA=setdiff(1:KK,TrainIdx_NA);
Y_Test_NA=Y_NA_K(TestIdx_NA);
level_Test_NA=KK_level(TestIdx_NA);


%%
%%%%%%%%%% Stage 1 %%%%%%%%%%%%
% In Stage 1, we take out random effects in the hidden process
% to determine the number of states and identify the important lags
% for each Group

%%% Assign Priors 
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
str_id = 8;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%   Final Model   %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    filename0=[folder,'Final_data_Seed_',num2str(seed),'_q_',num2str(qmax),'_d_',num2str(dmax),'.mat'];
    save(filename0,'model_str_best','idx_N','Train_Num_N','T','dmax','qmax','N1',...
        'level_N','level_Train_N','Y_Train_N','Y_Train_total_N','T_N','alpha_x',...
        'alpha00','Parameters_N','likelihood_type','Parameters_prior_N','pM',...
        'simsize','burnin','gap','pigamma','alpha_x0','alpha_x1','alpha00',...
        'alpha_a0','alpha_b0','alpha_a1','alpha_b1','omega','omega_a','omega_b',...
        'idx_NA','Train_Num_NA','level_Train_NA','Y_Train_NA','Y_Train_total_NA',...
        'T_NA','Parameters_NA','Parameters_prior_NA','level_Test_N',...
        'level_Test_NA','Y_Test_N','Y_Test_NA','Test_Num_N','Test_Num_NA')  
    

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




%% true generative model (Hfr.Ofr)
if classification==1
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
end

comp_time = toc;
results_final(1,7) = comp_time/60/60; % hours

%%
%%%%%%%%%%%%%%%%%%%%%%
%%% HOHMM learning %%%
%%%%%%%%%%%%%%%%%%%%%%
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Classification MHMM %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

comp_time = toc;
results_final(3,7) = comp_time/60/60; % hours

%%
%%%%%%%%%%%%%%%%%%%
%%%%% Results %%%%%
%%%%%%%%%%%%%%%%%%%
Model_Name={'Hf';'Hfr';'Hf.Of';'Hfr.Of';'Hf.Or';'Hfr.Or';...
   'Hf.Ofr';'Hfr.Ofr'};
%%% Testing set %%%
Model={Model_Name{str_id};'HOHMM';'MHMM'};
temp=table(Model);
results=array2table(results_final,'VariableNames',{'AUC','Accuracy','Sensivity','Specificity','Precision','F1-score','Comp_time'});
filename2=[folder,'Final_classification_test_Seed_',num2str(seed),'_q_',num2str(qmax),'_d_',num2str(dmax),'.xlsx'];
writetable([temp,results],filename2);

end



end

