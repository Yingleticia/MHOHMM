function [out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11,out12,...
    out13,out14,out15,out16]=MHOHMM_str_stage2_aoas(dmax,qmax,N1,Mact_L,simsize,burnin,gap,KK,...
    level_N,KK_level,C_total,CC_K,Xnew,Cnew,Y_K,Y_total,T_K,pigamma,...
    alpha_x0,alpha_x1,lambda_x,lambda_x0,alpha00,lambda00,alpha_a0,alpha_b0,...
    alpha_a1,alpha_b1,omega,omega_a,omega_b,Parameters,likelihood_type,...
    mu_r_K,beta,Parameters_prior,model_str)

% Model structure
% 1 -- fixed effect in hidden process;
% 2 -- random effect in hidden process;
% 3 -- fixed effect in observed process; 
% 4 -- random effect in observed process;
% 5 -- higher order in the hidden process.

if model_str(1)==0 % no fixed effects in hidden process
    level_N=1;
    alpha_x0=ones(level_N,1); % random effect in lambda
    alpha_x1=ones(level_N,1); % fixed effect in lambda
    KK_level=ones(1,length(KK_level));
end

if model_str(5)==0 % first order
    qmax=1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% group info
Num_L=zeros(level_N,1);
for ll=1:level_N
    Num_L(ll)=sum(KK_level==ll);
end
% Storage samples 
kgap=0;
lambda_x_storage=cell(level_N,floor((simsize-burnin)/gap));
lambda_x0_storage=zeros(dmax,level_N,floor((simsize-burnin)/gap));
alpha_x0_storage=ones(level_N,floor((simsize-burnin)/gap));
alpha_x1_storage=ones(level_N,floor((simsize-burnin)/gap));
omega_storage=zeros(level_N,2,floor((simsize-burnin)/gap));
pi_storage=cell(level_N,floor((simsize-burnin)/gap));
LogLikeli_storage=zeros(1,floor((simsize-burnin)/gap));

if strcmp(likelihood_type,'Normal')
    % Storage samples 
    mu_storage=zeros(dmax,floor((simsize-burnin)/gap));
    sigma_storage=zeros(dmax,floor((simsize-burnin)/gap));
    beta_storage=zeros(1,floor((simsize-burnin)/gap));
    Sigma_r_storage=zeros(1,floor((simsize-burnin)/gap));
    % initial values
    MuY_Mu0=Parameters_prior{1}; 
    MuY_SigmaSq0=Parameters_prior{2}; 
    kappa0=Parameters_prior{3}; 
    beta0=Parameters_prior{4}; 
    Mu_f=Parameters_prior{5};
    SigmaSq_f=Parameters_prior{6};
    kappa_r=Parameters_prior{7};
    beta_r=Parameters_prior{8};
    mu_r_all=zeros(sum(T_K),1); 
    mu_fix_all=zeros(sum(T_K),1);
    if model_str(4)==0
        mu_r_K=zeros(KK,1);
    end
    if model_str(3)==0
        beta=0;
    end
    for aa=1:KK
        mu_r_all((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)))=mu_r_K(aa);
        mu_fix_all((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)))=beta*KK_level(aa);
    end
elseif strcmp(likelihood_type,'MVN')
    % Storage samples 
    ObsDimension=size(Y_total,2);
    mu_storage=zeros(dmax,ObsDimension,floor((simsize-burnin)/gap));
    sigma_storage=zeros(ObsDimension,ObsDimension,dmax,floor((simsize-burnin)/gap));
    beta_storage=zeros(1,ObsDimension,floor((simsize-burnin)/gap));
    Sigma_r_storage=cell(1,floor((simsize-burnin)/gap));
    % initial values
    mu0=Parameters_prior{1}; 
    Sigma0=Parameters_prior{2};
    xi0=Parameters_prior{3}; 
    nu0=Parameters_prior{4}; 
    Mu_f=Parameters_prior{5};
    Sigma_f=Parameters_prior{6};
    Sigma0_r=Parameters_prior{7};
    nu_r=Parameters_prior{8};
    mu_r_all=zeros(sum(T_K),ObsDimension); 
    mu_fix_all=zeros(sum(T_K),ObsDimension);
    if model_str(4)==0
        mu_r_K=zeros(KK,ObsDimension);
    end
    if model_str(3)==0
        beta=zeros(1,ObsDimension);
    end
    for aa=1:KK
        mu_r_all((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)),:)=repmat(mu_r_K(aa,:),T_K(aa),1);
        mu_fix_all((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)),:)=repmat(beta*KK_level(aa),T_K(aa),1); 
    end
end

% allocation variables z, allocation prob. pi, and values of k_j
Tnew=T_K-qmax;
L00=ones(simsize,1); % number of states
State_select=zeros(simsize,dmax); % number of states
ind00_L=cell(1,level_N);
K00_L=cell(1,level_N);
p00_L=cell(1,level_N);
pigamma00_L=cell(1,level_N); % prior of pi_x
pi_L=cell(1,level_N);
for ll=1:level_N
    Mact=Mact_L{ll};
    MactFirstStage=Mact;
    aveM=mean(Mact(floor(N1/2):N1,:),1);
    MM0=round(aveM); 
    MProp=(MactFirstStage>1);
    MProp=mean(MProp(floor(N1/2):N1,:),1);
    ind00=find(MProp>0.5); 
    if sum(ind00)==0
       MM0(1)=2;
       ind00=1;
    end
    K00=MM0(ind00);         % k_{j}'s for the selected predictors
    p00=length(ind00);      % number of selected predictors
    ind00_L{ll}=ind00;
    K00_L{ll}=K00;
    p00_L{ll}=p00;
    pigamma00_L{ll}=pigamma(ind00);
    pi=zeros(p00,dmax,dmax);
    pi_L{ll}=pi;
end

z00_K=cell(KK,1);
x00_K=cell(KK,1);
for aa=1:KK
    ind00=ind00_L{KK_level(aa)};
    p00=p00_L{KK_level(aa)};
    K00=K00_L{KK_level(aa)};
    z00_aa=ones(Tnew(aa),p00);
    while length(unique(z00_aa))==1
          for j=1:p00             % initialize z_{ij}'s by randomly sampling from {1,...,k_{j}}
              z00_aa(:,j)=randsample(K00(j),Tnew(aa),true);
          end
    end
    z00_K{aa}=z00_aa;
    
    x00_aa=Xnew{aa};
    x00_aa=x00_aa(:,ind00);
    x00_K{aa}=x00_aa;
end

% transition weights
if model_str(2)==0 % no random effect in hidden process
    % omega
    omega=[0,1];
end 
omega_L=zeros(level_N,2);
for ll=1:level_N
    omega_L(ll,:)=omega;
end
% psi
psi_K=cell(1,KK);    
for aa=1:KK
    psi_aa=binornd(1,omega(2),[Tnew(aa),1]);
    psi_K{aa}=psi_aa;
end


lambda_aa=cell(1,KK);
for aa=1:KK
    lambda_aa{aa}=lambda_x{KK_level(aa)};
end

% level/group for all units at all time points
level_total=zeros(sum(T_K),1); 
level_total_new=zeros(sum(Tnew),1); 
psi_total=zeros(sum(Tnew),1); 
for aa=1:KK
    level_total((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)))=KK_level(aa);
    level_total_new((sum(Tnew(1:(aa-1)))+1):sum(Tnew(1:aa)))=KK_level(aa);
    psi_total((sum(Tnew(1:(aa-1)))+1):sum(Tnew(1:aa)))=psi_K{aa};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% iteration start %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%
for k=1:simsize
%     tic
    %%%%%%% pi %%%%%%
    for ll=1:level_N
        p00=p00_L{ll};
        K00=K00_L{ll};
        idx_level=find(KK_level==ll);
        cp=zeros(p00,dmax,dmax);   % counts for pi,first=j,second=value of x,third=value of z
        for aa=1:Num_L(ll)
            x00_aa=x00_K{idx_level(aa)};
            z00_aa=z00_K{idx_level(aa)};
            for t=1:Tnew(idx_level(aa))
                for j=1:p00
                    cp(j,x00_aa(t,j),z00_aa(t,j))=cp(j,x00_aa(t,j),z00_aa(t,j))+1;
                end
            end
        end
        pi=pi_L{ll};
        pigamma00=pigamma00_L{ll};
        for j=1:p00
            for s=1:dmax
                rr=gamrnd(cp(j,s,1:K00(j))+pigamma00(j),1);
                pi(j,s,1:K00(j))=rr/sum(rr);	% s=x_{j}, only the first k_{j} elements are updated, rest are left at zero
            end
            % switch labels 
            % in the reshaped d_{j}*k_{j} matrix below, the rows correspond to the values of x_{j} and the columns to the values of h_{j}, the sum across each row is one 
            [~, qq2]=sort(sum(reshape(pi(j,:,1:K00(j)),dmax,K00(j)),1),'descend');
            for s=1:dmax
                pi(j,s,1:K00(j))=pi(j,s,qq2);    % column labels h_{j}'s are switched
            end
                        
            for aa=1:Num_L(ll)
                z00_aa=z00_K{idx_level(aa)};
                for t=1:Tnew(idx_level(aa))
                    z00_aa(t,j)=find(qq2==z00_aa(t,j));	% z_{t,j}'s are switched
                end 
                z00_K{idx_level(aa)}=z00_aa;
            end
        end
        pi_L{ll}=pi;
    end   
    
   
    
    
    %%%%%% lambda %%%%%%
    m_0=zeros(dmax,1);% counting number for lambda00
    for ll=1:level_N
        K00=K00_L{ll};
        idx_level=find(KK_level==ll);
        
        %%%%%% fixed term: lambda_x %%%%%%
        z00_aa=z00_K{idx_level(1)};
        psi_aa=psi_K{idx_level(1)};
        Cnew_aa=Cnew{idx_level(1)};
        if sum(psi_aa==1)~=0 % in case all psi_aa=0
            z00_aa=z00_aa(psi_aa==1,:); % psi=1: fixed term
            Cnew_aa=Cnew_aa(psi_aa==1); % psi=1: fixed term
            clT=tensor(zeros([dmax,K00]),[dmax,K00]);
            [z0,m]=unique(sortrows([Cnew_aa z00_aa]),'rows','legacy');
            clT(z0)=clT(z0)+m-[0;m(1:(end-1))];
            clTdata_total=tenmat(clT,1);
        else
            clT=tensor(zeros([dmax,K00]),[dmax,K00]);
            clTdata_total=tenmat(clT,1);
        end
        
        for aa=2:Num_L(ll)
            z00_aa=z00_K{idx_level(aa)};
            psi_aa=psi_K{idx_level(aa)};
            Cnew_aa=Cnew{idx_level(aa)};
            if sum(psi_aa==1)~=0 % in case all psi_aa=0
                z00_aa=z00_aa(psi_aa==1,:); % psi=1: fixed term
                Cnew_aa=Cnew_aa(psi_aa==1); % psi=1: fixed term
                clT=tensor(zeros([dmax,K00]),[dmax,K00]);
                [z0,m]=unique(sortrows([Cnew_aa z00_aa]),'rows','legacy');
                clT(z0)=clT(z0)+m-[0;m(1:(end-1))];
                clTdata=tenmat(clT,1);
                clTdata_total=clTdata_total+clTdata;
            else
                clT=tensor(zeros([dmax,K00]),[dmax,K00]);
                clTdata=tenmat(clT,1);
                clTdata_total=clTdata_total+clTdata;
            end
        end
        sz=size(clTdata_total);
        n_x=clTdata_total.data;    
        lambdamat=zeros(dmax,sz(2));
        for j=1:sz(2)
            lambdamat(:,j)=gamrnd(n_x(:,j)+alpha_x1(ll)*lambda_x0(:,ll),1);
            while sum(lambdamat(:,j))==0
                  lambdamat(:,j)=gamrnd(n_x(:,j)+alpha_x1(ll)*lambda_x0(:,ll),1);
            end
            lambdamat(:,j)=lambdamat(:,j)/sum(lambdamat(:,j));
        end
        lambda_x{ll}=tensor(lambdamat,[dmax,K00]);
       
        %%%%%% alpha_x1 %%%%%%
        v_x=zeros(dmax,sz(2)); % v_x(c|h_1,...,h_qx)
        for i=1:dmax
            for m=1:sz(2)
                prob=alpha_x1(ll)*lambda_x0(i,ll)./((1:n_x(i,m))-1+alpha_x1(ll)*lambda_x0(i,ll));
                v_x(i,m)=v_x(i,m)+sum(binornd(ones(1,n_x(i,m)),prob));
            end
        end                
        mx_1=sum(v_x,'all');
        n_x_colsums=sum(n_x,1); 
        sx_1=binornd(ones(1,sz(2)),n_x_colsums./(n_x_colsums+alpha_x1(ll)));
        rx_1=betarnd(alpha_x1(ll)+1,n_x_colsums);
        alpha_x1(ll)=gamrnd(alpha_a1+mx_1-sum(sx_1),1/(alpha_b1-sum(log(rx_1))));
               
        
        if model_str(2)==1 % random effect in hidden process
            %%%%%% random term: lambda_aa %%%%%%
            % several counting numbers
            v_a=zeros(dmax,sz(2)); % cumulative v_a(c|h_1,...,h_qx)
            sx_0=zeros(1,sz(2)); % cumulative s_a(h_1,...,h_qx)
            log_rx_0=zeros(1,sz(2)); % cumulative log r_a(h_1,...,h_qx)
            for aa=1:Num_L(ll)
                z00_aa=z00_K{idx_level(aa)};
                psi_aa=psi_K{idx_level(aa)};
                Cnew_aa=Cnew{idx_level(aa)};
                if sum(psi_aa==0)~=0
                    z00_aa=z00_aa(psi_aa==0,:); % psi=0: random term
                    Cnew_aa=Cnew_aa(psi_aa==0); % psi=0: random term
                    clT=tensor(zeros([dmax,K00]),[dmax,K00]);
                    [z0,m]=unique(sortrows([Cnew_aa z00_aa]),'rows','legacy');
                    clT(z0)=clT(z0)+m-[0;m(1:(end-1))];
                    clTdata=tenmat(clT,1);
                    sz=size(clTdata);
                    n_aa=clTdata.data; % transition counting for unit a
                    lambdamat=zeros(dmax,sz(2));
                    for j=1:sz(2)
                        lambdamat(:,j)=gamrnd(n_aa(:,j)+alpha_x0(ll)*lambda_x0(:,ll),1);
                        while sum(lambdamat(:,j))==0
                            lambdamat(:,j)=gamrnd(n_aa(:,j)+alpha_x0(ll)*lambda_x0(:,ll),1);
                        end
                        lambdamat(:,j)=lambdamat(:,j)/sum(lambdamat(:,j));
                    end
                    lambda_aa{idx_level(aa)}=tensor(lambdamat,[dmax,K00]);

                    v_a_temp=zeros(dmax,sz(2)); % v_a(c|h_1,...,h_qx)
                    for i=1:dmax
                        for m=1:sz(2)
                            prob=alpha_x0(ll)*lambda_x0(i,ll)./((1:n_aa(i,m))-1+alpha_x0(ll)*lambda_x0(i,ll));
                            v_a_temp(i,m)=v_a_temp(i,m)+sum(binornd(ones(1,n_aa(i,m)),prob));
                        end
                    end
                    v_a=v_a+v_a_temp; % cumulative
                    n_aa_colsums=sum(n_aa,1);
                    sx_0=sx_0+binornd(ones(1,sz(2)),n_aa_colsums./(n_aa_colsums+alpha_x0(ll)));
                    log_rx_0=log_rx_0+log(betarnd(alpha_x0(ll)+1,n_aa_colsums));               
                end
            end
            %%%%%% alpha_x0 %%%%%%
            mx_0=sum(v_a,'all');
            alpha_x0(ll)=gamrnd(alpha_a0+mx_0-sum(sx_0),1/(alpha_b0-sum(log_rx_0)));
        elseif model_str(2)==0 % no random effect in hidden process
            v_a=zeros(dmax,sz(2));
        end
        
        %%%%%% lambda_x0 %%%%%%
        m_x=sum(v_x,2)+sum(v_a,2);
        lambda_temp=gamrnd(m_x+alpha00*lambda00,1);
        lambda_temp=lambda_temp/sum(lambda_temp);
        lambda_temp(lambda_temp==0)=10^(-5);    % For numerical reasons
        lambda_x0(:,ll)=lambda_temp;
        m_0=m_0+m_x;
        
    end
    
    %%%%%% lambda00 %%%%%%
    lambda00=gamrnd(m_0+1/dmax,1);
    lambda00=lambda00/sum(lambda00);
    lambda00(lambda00==0)=10^(-5);    % For numerical reasons
    
    if model_str(2)==1 % random effect in hidden process
        
        %%%%%% psi %%%%%%
       for aa=1:KK
            Cnew_aa=Cnew{aa};
            z00_aa=z00_K{aa};
            omega_xx=omega_L(KK_level(aa),:);
            lambda_aa_1=lambda_x{KK_level(aa)}; % fixed lambda
            lambda_aa_0=lambda_aa{aa}; % random lambda
            prob = omega_xx.*[lambda_aa_0([Cnew_aa,z00_aa]) lambda_aa_1([Cnew_aa,z00_aa])];
            prob=bsxfun(@rdivide,prob,sum(prob,2));
            psi_aa=binornd(ones(1,Tnew(aa)),prob(:,2)')';
            psi_K{aa}=psi_aa;
            psi_total((sum(Tnew(1:(aa-1)))+1):sum(Tnew(1:aa)))=psi_aa;
       end 
       
        %%%%%% weights: omega %%%%%%
        for ll=1:level_N
            weight_N=zeros(1,2);
            weight_N(1)=sum(psi_total(level_total_new==ll)==0); % 0: random effect;
            weight_N(2)=sum(psi_total(level_total_new==ll)==1); % 1: fixed effect;
            omega_x=zeros(1,2);
            omega_x(2)=betarnd(omega_a+weight_N(2),omega_b+weight_N(1));
            omega_x(1)=1-omega_x(2);
            omega_L(ll,:)=omega_x;
        end     
    end
    
    
    %%%%%% z %%%%%%
    for aa=1:KK
        x00_aa=x00_K{aa};
        z00_aa=z00_K{aa};
        Cnew_aa=Cnew{aa};
        p00=p00_L{KK_level(aa)};
        K00=K00_L{KK_level(aa)};
        pi=pi_L{KK_level(aa)};
        lambda_fix=lambda_x{KK_level(aa)}; % fixed lambda
        
        prob2_hh = cell(1,p00);
        ii = ones(Tnew(aa),1);
        z00_aa_ori = z00_aa;
        
        if model_str(2)==0 % no random effect in hidden process
            for j=1:p00
                prob_hhh = zeros(Tnew(aa),K00(j));
                for h=1:K00(j)
                    II = [Cnew_aa,z00_aa(:,1:(j-1)),h*ii,z00_aa_ori(:,(j+1):p00)];
                    prob_hhh(:,h)=pi(j,x00_aa(:,j),h)'.*lambda_fix(II);
                end
                prob2_hh{j} = prob_hhh./sum(prob_hhh,2);
                z00_aa(:,j)=sum(mnrnd(1,prob2_hh{j}).*(1:K00(j)),2);
            end 
        elseif model_str(2)==1 % random effect in hidden process
            psi_aa=psi_K{aa};
            lambda_rand=lambda_aa{aa}; % random lambda
            JJ = psi_aa~=0; % fixed effect
            for j=1:p00
                prob_hh = zeros(Tnew(aa),K00(j));
                prob_hhh = zeros(Tnew(aa),K00(j));
                for h=1:K00(j)
                    II = [Cnew_aa,z00_aa(:,1:(j-1)),h*ii,z00_aa_ori(:,(j+1):p00)];
                    prob_hh(:,h)=pi(j,x00_aa(:,j),h)'.*lambda_rand(II);
                    prob_hhh(:,h)=pi(j,x00_aa(:,j),h)'.*lambda_fix(II);
                end
                prob2_hh{j} = prob_hh./sum(prob_hh,2);
                prob2_hh{j}(JJ,:) = prob_hhh(JJ,:)./sum(prob_hhh(JJ,:),2);
                z00_aa(:,j)=sum(mnrnd(1,prob2_hh{j}).*(1:K00(j)),2);
            end
        end
        z00_K{aa}=z00_aa;         
    end
    
    %%%%%% C %%%%%%
%     if (k>2)
    if (k>100)
        for aa=1:KK
            Y_aa=Y_K{aa};
            Ynew_aa=Y_aa((qmax+1):end,:);
            z00_aa=z00_K{aa};
            Cnew_aa=Cnew{aa};
            C_aa=CC_K{aa};
            ind00=ind00_L{KK_level(aa)};
            p00=p00_L{KK_level(aa)};
            pi=pi_L{KK_level(aa)};
            
            for tt=1:qmax
                ProbVect=zeros(dmax,1);
                ProbVect(unique(Cnew_aa))=likelihood_fn_str(Y_aa(tt,:),Parameters,...
                    likelihood_type,beta*KK_level(aa),mu_r_K(aa),unique(Cnew_aa));
                if sum(ProbVect)>0
                   ProbVect=ProbVect./sum(ProbVect);
                   C_aa(tt)=randsample(dmax,1,'true',ProbVect);
                end
            end
            
            % Cnew
            likelihood=likelihood_fn_str(Ynew_aa,Parameters,likelihood_type,...
                beta*KK_level(aa),mu_r_K(aa,:));
            ProbVect=ones(Tnew(aa),dmax);
            lambda_fix=lambda_x{KK_level(aa)}; % fixed lambda  
            for kk=1:dmax
                
                if model_str(2)==0 % no random effect in hidden process
                   lambda_sub = lambda_fix([kk*ones(Tnew(aa),1),z00_aa]); 
                elseif model_str(2)==1 % random effect in hidden process
                    psi_aa=psi_K{aa};
                    lambda_rand=lambda_aa{aa}; % random lambda
                    lambda_sub = lambda_rand([kk*ones(Tnew(aa),1),z00_aa]);
                    lambda_fix_sub = lambda_fix([kk*ones(Tnew(aa),1),z00_aa]);
                    lambda_sub(psi_aa~=0,:) = lambda_fix_sub(psi_aa~=0,:);
                end
                
                II_z00 = (1:(Tnew(aa)-qmax))'+ind00; %(1:(Tnew(aa)-qmax))'
                hhh = z00_aa(II_z00',:);
                [nrh, nch] = size(hhh);
                iii = sub2ind([nrh, nch],(1:nrh)',repmat((1:nch)',nrh/nch,1));
                hhh = hhh(iii);
                pi_sub = ones(Tnew(aa),1);
                temp = reshape(pi(:,kk,hhh),p00,[]);
                temp = temp';
                temp = prod(reshape(temp(iii),p00,[]),1);
                pi_sub(1:(Tnew(aa)-qmax),:) = temp';
                likelihood_sub = likelihood(:,kk);
                ProbVect(:,kk) = lambda_sub.*pi_sub.*likelihood_sub;
                
            end
        
            II = sum(ProbVect,2)>0;
            ProbVect = ProbVect./sum(ProbVect,2);
            for tt=1:Tnew(aa)
                if II(tt)
                    Cnew_aa(tt) = randsample(1:dmax,1,'true',ProbVect(tt,:));
                end
            end
            C_aa((qmax+1):end)=Cnew_aa;
            CC_K{aa}=C_aa;
            Cnew{aa}=Cnew_aa;
            Xnew_aa = C_aa(((qmax+1):T_K(aa))'-(1:qmax));
            x00_K{aa}=Xnew_aa(:,ind00);
            C_total((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)))=C_aa;  
        end
    end 
    L00(k)=length(unique(C_total)); % required number of states 
    State_select(k,:)=ismember(1:dmax,unique(C_total));
    
    %%%%%% emission distribution %%%%%%
%     if (k>2)
    if (k>100)
        % MuY and SigmaSqY
        MuY=Parameters{1};
        SigmaY=Parameters{2};
        tbl=zeros(dmax,3);
        tbl(1:max(C_total),:)=tabulate(C_total);
        
        if strcmp(likelihood_type,'Normal')
            % MuY and SigmaY
            for kk=1:dmax
                temp=find(C_total==kk);
                MuY_SigmaSq=1/(1/MuY_SigmaSq0+tbl(kk,2)/SigmaY(kk));
                MuY_Mu=MuY_SigmaSq*(MuY_Mu0/MuY_SigmaSq0+sum(Y_total(temp)-...
                    mu_r_all(temp)-mu_fix_all(temp))/SigmaY(kk));
                MuY(kk)=normrnd(MuY_Mu,sqrt(MuY_SigmaSq),1);
                SigmaY(kk)=1./gamrnd(kappa0+tbl(kk,2)/2,1/(beta0+...
                    sum((Y_total(temp)-mu_r_all(temp)-mu_fix_all(temp)-MuY(kk)).^2)/2),1);
            end
            Parameters{1}=MuY;
            Parameters{2}=SigmaY;
            
            % random effect in observed process
            if model_str(4)==1 
                % Sigma_r
                Sigma_r=1./gamrnd(kappa_r+KK/2,1/(beta_r+sum((mu_r_K).^2)/2),1);
                % mu_r_K
                for aa=1:KK
                    C_aa=CC_K{aa};
                    Y_aa=Y_K{aa};
                    tbl_aa=zeros(dmax,3);
                    tbl_aa(1:max(C_aa),:)=tabulate(C_aa);
                    Mu_r_SigmaSq=1/(1/Sigma_r+sum(tbl_aa(:,2)./SigmaY));
                    temp=sum((Y_aa-MuY(C_aa)-beta*KK_level(aa))./SigmaY(C_aa));
                    Mu_r_Mu=Mu_r_SigmaSq*temp;
                    mu_r_K(aa)=normrnd(Mu_r_Mu,sqrt(Mu_r_SigmaSq),1);
                end
                % random effect for all units at each time point
                for aa=1:KK
                    mu_r_all((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)))=mu_r_K(aa);
                end
            end
            
            % fixed effect in observed process
            if model_str(3)==1 
                % beta
                beta_Sigma_temp=zeros(dmax,level_N);
                data_temp=zeros(1,level_N);
                for ll=1:level_N
                    C_total_ll=C_total(level_total==ll);
                    Y_total_ll=Y_total(level_total==ll);
                    mu_r_all_ll=mu_r_all(level_total==ll);
                    temp=tabulate(C_total_ll);
                    beta_Sigma_temp(1:max(C_total_ll),ll)=temp(:,2)*(ll^2)...
                        ./SigmaY(1:max(C_total_ll));
                    data_temp(ll)=sum((Y_total_ll-MuY(C_total_ll)-mu_r_all_ll)*ll./SigmaY(C_total_ll));
                end
                beta_Sigma=1/(1/SigmaSq_f+sum(beta_Sigma_temp,'all'));
                beta_Mu=beta_Sigma*(Mu_f/SigmaSq_f+sum(data_temp));
                beta=normrnd(beta_Mu,sqrt(beta_Sigma),1);
                % fixed effect for all units at each time point
                for aa=1:KK
                    mu_fix_all((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)))=beta*KK_level(aa);
                end
            end
        
        elseif strcmp(likelihood_type,'MVN')
            % MuY and SigmaY
            for kk=1:dmax
                temp=find(C_total==kk);
                if tbl(kk,2)>1
                    SigmaN=Sigma0+(tbl(kk,2)-1)*cov(Y_total(temp,:)-...
                        mu_r_all(temp,:)-mu_fix_all(temp,:))+...
                         (xi0*tbl(kk,2)/(xi0+tbl(kk,2)))*...
                         (mean(Y_total(temp,:)-mu_r_all(temp,:)-...
                         mu_fix_all(temp,:))-mu0)'*(mean(Y_total(temp,:)-...
                         mu_r_all(temp,:)-mu_fix_all(temp,:))-mu0);
                     muN=(xi0*mu0+tbl(kk,2)*mean(Y_total(temp,:)-...
                         mu_r_all(temp,:)-mu_fix_all(temp,:)))/(xi0+tbl(kk,2));
                     xiN=xi0+tbl(kk,2);
                     nuN=nu0+tbl(kk,2);
                     SigmaY(:,:,kk)=iwishrnd(SigmaN,nuN);
                     MuY(kk,:)=mvnrnd(muN,SigmaY(:,:,kk)/xiN,1);
                elseif tbl(kk,2)==1
                    SigmaN=Sigma0+(xi0/(xi0+1))*(Y_total(temp,:)-...
                        mu_r_all(temp,:)-mu_fix_all(temp,:)-mu0)'*...
                        (Y_total(temp,:)-mu_r_all(temp,:)-mu_fix_all(temp,:)-mu0);
                     muN=(xi0*mu0+Y_total(temp,:)-mu_r_all(temp,:)-...
                         mu_fix_all(temp,:))/(xi0+1);
                     xiN=xi0+tbl(kk,2);
                     nuN=nu0+tbl(kk,2);
                     SigmaY(:,:,kk)=iwishrnd(SigmaN,nuN);
                     MuY(kk,:)=mvnrnd(muN,SigmaY(:,:,kk)/xiN,1);
                end
            end
            Parameters{1}=MuY;
            Parameters{2}=SigmaY;
            
            % random effect in observed process
            if model_str(4)==1 
                % Sigma_r
                nu_xN=KK+nu_r;
                Sigma_xN=Sigma0_r+mu_r_K'*mu_r_K;
                Sigma_r=iwishrnd(Sigma_xN,nu_xN);
                % mu_r_K
                for aa=1:KK
                    C_aa=CC_K{aa};
                    Y_aa=Y_K{aa};
                    tbl_aa=zeros(dmax,3);
                    tbl_aa(1:max(C_aa),:)=tabulate(C_aa);
                    temp_Sigma=inv(Sigma_r);
                    temp_Mu=zeros(size(MuY));
                    for mm=1:dmax
                        if tbl_aa(mm,2)>1
                            temp_Sigma=temp_Sigma+inv(SigmaY(:,:,mm)/tbl_aa(mm,2));
                            temp_Mu(mm,:)=(SigmaY(:,:,mm)\((sum(Y_aa(C_aa==mm,:)...
                                -MuY(mm,:)-beta*KK_level(aa)))'))';
                        elseif tbl_aa(mm,2)==1
                            temp_Sigma=temp_Sigma+inv(SigmaY(:,:,mm)/tbl_aa(mm,2));
                            temp_Mu(mm,:)=(SigmaY(:,:,mm)\((Y_aa(C_aa==mm,:)-...
                                MuY(mm,:)-beta*KK_level(aa))'))';
                        end
                    end
                    Mu_r=temp_Sigma\((sum(temp_Mu))');
                    mu_r_K(aa,:)=mvnrnd(Mu_r',inv(temp_Sigma),1);
                end
                % random effect for all units at each time point
                for aa=1:KK
                    mu_r_all((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)),:)=repmat(mu_r_K(aa,:),T_K(aa),1);
                end
            end
            
            % fixed effect in observed process
            if model_str(3)==1
                % beta
                beta_Sigma=inv(Sigma_f);
                data_temp=zeros(size(MuY));
                for ll=1:level_N
                    C_total_ll=C_total(level_total==ll);
                    Y_total_ll=Y_total(level_total==ll);
                    mu_r_all_ll=mu_r_all(level_total==ll);
                    tbl_ll=zeros(dmax,3);
                    tbl_ll(1:max(C_total_ll),:)=tabulate(C_total_ll);
                    temp_Mu=zeros(size(MuY));
                    for mm=1:dmax
                        if tbl_ll(mm,2)>1
                            beta_Sigma=beta_Sigma+inv(SigmaY(:,:,mm)/(ll^2)/tbl_ll(mm,2));
                            temp_Mu(mm,:)=(SigmaY(:,:,mm)\((sum(Y_total_ll(C_total_ll==mm,:)...
                                -MuY(mm,:)-mu_r_all_ll(C_total_ll==mm,:))*ll)'))';
                        elseif tbl_ll(mm,2)==1
                            beta_Sigma=beta_Sigma+inv(SigmaY(:,:,mm)/(ll^2)/tbl_ll(mm,2));
                            temp_Mu(mm,:)=(SigmaY(:,:,mm)\(((Y_total_ll(C_total_ll==mm,:)...
                                -MuY(mm,:)-mu_r_all_ll(C_total_ll==mm,:))*ll)'))';
                        end
                    end
                    data_temp=data_temp+temp_Mu;
                end
                beta_Mu=beta_Sigma\((sum(data_temp))');
                beta=mvnrnd(beta_Mu',inv(beta_Sigma),1);
                % fixed effect for all units at each time point
                for aa=1:KK
                    mu_fix_all((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)),:)=repmat(beta*KK_level(aa),T_K(aa),1);
                end
            end
            
        end
     
    end
    %%%%%%%%%%%%%%%%%% 
    %%%% storage %%%%%
    %%%%%%%%%%%%%%%%%%
    if k>burnin && mod(k,gap)==0
        
        alpha_x0_storage(:,kgap+1)=alpha_x0;
        alpha_x1_storage(:,kgap+1)=alpha_x1;
        
       if strcmp(likelihood_type,'Normal')
          mu_storage(:,kgap+1)=MuY;
          sigma_storage(:,kgap+1)=SigmaY;
          beta_storage(kgap+1)=beta;
          if model_str(4)==1
              Sigma_r_storage(kgap+1)=Sigma_r;
          end
          LogLikeli_storage(kgap+1)=sum(log(normpdf(Y_total,MuY(C_total)...
              +mu_r_all+mu_fix_all,sqrt(SigmaY(C_total)))));
       elseif strcmp(likelihood_type,'MVN')
          mu_storage(:,:,kgap+1)=MuY; 
          sigma_storage(:,:,:,kgap+1)=SigmaY;
          beta_storage(:,:,kgap+1)=beta;
          if model_str(4)==1
              Sigma_r_storage{kgap+1}=Sigma_r;
          end
          LogLikeli_storage(kgap+1)=sum(log(mvnpdf(Y_total,MuY(C_total,:)...
              +mu_r_all+mu_fix_all,SigmaY(:,:,C_total))));
       end 
       
       for ll=1:level_N
           pi_storage{ll,kgap+1}=pi_L{ll};
           lambda_x_storage{ll,kgap+1}=lambda_x{ll};
           lambda_x0_storage(:,ll,kgap+1)=lambda_x0(:,ll);  
           omega_storage(:,:,kgap+1)=omega_L;
       end            
       kgap=kgap+1;
    end
    
    %%%%%%%%%%%%%%%%%% 
    %%% print info %%%
    %%%%%%%%%%%%%%%%%%
    fprintf('k = %i, states = [%i], ',k,L00(k));
    for ll=1:level_N
        p00=p00_L{ll};
        ind00=ind00_L{ll};
        K00=K00_L{ll};
        fprintf('%i predictors = {',p00);
        for i=1:p00
            fprintf(' C(t-%i)(%i)',ind00(i),K00(i));
        end
        fprintf(' }.');
    end
     fprintf(' \n');
%      toc
end

%%%% output %%%%
out1=lambda_x_storage;
out2=lambda_x0_storage;
out3=alpha_x0_storage;
out4=alpha_x1_storage;
out5=omega_storage;
out6=pi_storage;
out7=mu_storage;
out8=sigma_storage;
out9=beta_storage;
out10=Sigma_r_storage;
out11=L00;
out12=State_select;
out13=p00_L;
out14=K00_L;
out15=ind00_L;
out16=LogLikeli_storage;

end