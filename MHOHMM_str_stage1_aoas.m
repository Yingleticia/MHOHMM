function [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11 ...
    ]=MHOHMM_str_stage1_aoas(dmax,qmax,N1,KK,level_N,KK_level,CC_K,Y_K,...
    Y_total,T_K,alpha_x,alpha00,Parameters,beta,...
    likelihood_type,Parameters_prior,pM,model_str)
% Model structure
% 1 -- fixed effect in hidden process;
% 2 -- random effect in hidden process;
% 3 -- fixed effect in observed process; 
% 4 -- random effect in observed process;
% 5 -- higher order in the hidden process.

if model_str(1)==0 % no fixed effects in hidden process
    level_N=1;
    KK_level=ones(1,length(KK_level));
end

if model_str(5)==0 % first order
    qmax=1;
end

Tnew=T_K-qmax;
Cnew=cell(KK,1);
Ynew=cell(KK,1);
Xnew=cell(KK,1);
for aa=1:KK
    A=CC_K{aa};
    Cnew{aa}=A((qmax+1):end);
    B=Y_K{aa};
    Ynew{aa}=B((qmax+1):end,:);
    C=zeros(Tnew(aa),qmax); 
    for j=1:qmax                            
       C(:,j)=A((qmax+1-j):(T_K(aa)-j));  
    end  
    Xnew{aa}=C;
end

L=ones(N1+1,1);
% Parameters for different levels
M_L=cell(1,level_N);
Mact_L=cell(1,level_N);
G_L=cell(1,level_N);

Z_K=cell(KK,1);
for ll=1:level_N
    M=ones(N1+1,qmax);
    Mact=ones(N1,qmax);
    G=ones(qmax,dmax);
    
    idx_level=find(KK_level==ll);
    temp=true(1,length(idx_level));
    while sum(temp)>0
        G(1,:)=randsample(2,dmax,'true',[0.5,0.5]);
        for aa=1:length(idx_level)
            z_aa=ones(Tnew(idx_level(aa)),qmax); 
            Xnew_aa=Xnew{idx_level(aa)};
            z_aa(:,1)=G(1,Xnew_aa(:,1));
            temp(aa)=(length(unique(z_aa(:,1)))==1);
            Z_K{idx_level(aa)}=z_aa;
        end
    end
    Mact(1,1)=2;M(1,1)=2;   % Start with k_{1}=2
    M_L{ll}=M;
    Mact_L{ll}=Mact;
    G_L{ll}=G;
end


log0=zeros(N1,1);
cM=zeros(qmax,dmax);
for j=1:qmax                                % There are (2^r-2)/2 ways to split r objects into two non-empty groups.
    cM(j,1:dmax)=(2.^(1:dmax)-2)/2;         % (Consider 2 cells and r objects, subtract 2 for the two cases in which 
end

lambda_x0=repmat(1/dmax,dmax,level_N); % state preference for each group/level
lambda00=repmat(1/dmax,dmax,1);
temperature0=1000;

% level/group for all units at all time points
level_total=zeros(sum(T_K),1); 
for aa=1:KK
    level_total((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)))=KK_level(aa);
end

% Priors for observed process
if strcmp(likelihood_type,'Normal')
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
    if model_str(4)==0 % no random effect in obs process
        mu_r_K=zeros(KK,1);
    elseif model_str(4)==1 % random effect in obs process
        mu_r_K=normrnd(0,0.1,KK,1);
    end
    if model_str(3)==0 % no fixed effect in obs process
        beta=0;
    end
    for aa=1:KK
        mu_r_all((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)))=mu_r_K(aa);
        mu_fix_all((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)))=beta*KK_level(aa);
    end
elseif strcmp(likelihood_type,'MVN')
    ObsDimension=size(Y_total,2);
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
    if model_str(4)==0 % no random effect in obs process
        mu_r_K=zeros(KK,ObsDimension);
    elseif model_str(4)==1 % random effect in obs process
        mu_r_K=mvnrnd(zeros(1,ObsDimension),Sigma0_r,KK,1); 
    end
    if model_str(3)==0 % no fixed effect in obs process
        beta=zeros(1,ObsDimension);
    elseif model_str(3)==1 % fixed effect in obs process  
        beta=beta*ones(1,ObsDimension);
    end
    for aa=1:KK
        mu_r_all((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)),:)=repmat(mu_r_K(aa,:),T_K(aa),1);
        mu_fix_all((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)),:)=repmat(beta*KK_level(aa),T_K(aa),1); 
    end
end
%%%%%%%%%%%%%%%%
%%% Sampling %%%
%%%%%%%%%%%%%%%%
%%
for k=1:N1
    M00_L=cell(1,level_N);
    ind00_L=cell(1,level_N);
    K00_L=cell(1,level_N);
    p00_L=cell(1,level_N);
    
    for ll=1:level_N
        M=M_L{ll};
        M00=M(k,:);          % initiate M00={k_{1},...,k_{p}}, current values of k_{j}'s for the kth iteration
        ind00=find(M00>1);   % selected predictors for which k_{j}>1
        if isempty(ind00)
            ind00=1;
        end
        K00=M00(ind00);      % k_{j}'s for the selected predictors
        p00=length(ind00);   % number of selected predictors
        M00_L{ll}=M00;
        ind00_L{ll}=ind00;
        K00_L{ll}=K00;
        p00_L{ll}=p00;
    end
    
    % lambda_x & lambda_x0 & lambda00
    lambda_x=cell(1,level_N);
    mmm=zeros(dmax,1);
    
    for ll=1:level_N
        ind00=ind00_L{ll};
        K00=K00_L{ll};
        p00=p00_L{ll};
        idx_level=find(KK_level==ll); % units in each group
        
        z00=Z_K{idx_level(1)};
        z00=z00(:,ind00);
        clT=tensor(zeros([dmax,K00]),[dmax,K00]);
        [z0,m]=unique(sortrows([Cnew{idx_level(1)} z00]),'rows','legacy');
        clT(z0)=clT(z0)+m-[0;m(1:(end-1))];
        clTdata=tenmat(clT,1);
        clTdata_total=clTdata;
        for aa=2:length(idx_level)
            z00=Z_K{idx_level(aa)};
            z00=z00(:,ind00);
            clT=tensor(zeros([dmax,K00]),[dmax,K00]);
            [z0,m]=unique(sortrows([Cnew{idx_level(aa)} z00]),'rows','legacy');
            clT(z0)=clT(z0)+m-[0;m(1:(end-1))];
            clTdata=tenmat(clT,1);
            clTdata_total=clTdata_total+clTdata;
        end
        sz=size(clTdata_total);
        a=zeros(dmax,sz(2));
        for i=1:dmax
            a(i,:)=tenmat(tensor(tenmat(clTdata_total(i,:),[],1:p00,K00)),[],'t'); % tenmat: store tensor as a matrix, e.g., a is 3*4*5, then tenmat(a,1) -> 3*(4*5) matrix
        end
        % lambda_x
        lambdamat=zeros(dmax,sz(2));
        for j=1:sz(2)
            lambdamat(:,j)=gamrnd(a(:,j)+alpha_x*lambda_x0(:,ll),1);
            lambdamat(:,j)=lambdamat(:,j)/sum(lambdamat(:,j));
        end
        lambda_x{ll}=tensor(lambdamat,[dmax,K00]);
        
        mmat=zeros(dmax,sz(2));
        for i=1:dmax
            for m=1:sz(2)
%                 prob=alpha_x*lambdamat(i,m)./((1:a(i,m))-1+alpha_x*lambdamat(i,m));
                prob=alpha_x*lambda_x0(i,ll)./((1:a(i,m))-1+alpha_x*lambda_x0(i,ll));
                mmat(i,m)=mmat(i,m)+sum(binornd(ones(1,a(i,m)),prob));
            end
        end
        lambda_temp=gamrnd(sum(mmat,2)+alpha00*lambda00,1);
        lambda_temp=lambda_temp/sum(lambda_temp);
        lambda_temp(lambda_temp==0)=10^(-5);    % For numerical reasons
        lambda_x0(:,ll)=lambda_temp;
        mmm=mmm+sum(mmat,2);
    end
    
    % lambda00
    lambda00=gamrnd(mmm+1/(1*dmax),1);
    lambda00=lambda00/sum(lambda00);
    lambda00(lambda00==0)=10^(-5);    % For numerical reasons
   
    % C and Cnew given z ignoring G
    L00_K=zeros(KK,1);
    CC_K_prop=CC_K;
    Z_K_prop=Z_K;
    Cnewprop=Cnew;
    Xnewprop=Xnew;
    
    for aa=1:KK
        ind00=ind00_L{KK_level(aa)};
        G=G_L{KK_level(aa)};
        lambda=lambda_x{KK_level(aa)};
        
        Y_aa=Y_K{aa};
        Ynew_aa=Ynew{aa};
        Cprop_aa=CC_K_prop{aa};
        Cnew_aa=Cnew{aa};
        z00_aa=Z_K{aa};
        z00_aa=z00_aa(:,ind00);
        Cnewprop_aa=Cnewprop{aa};
        Xnewprop_aa=Xnewprop{aa};
        Z_K_prop_aa=Z_K_prop{aa};
        % C
        for tt=1:qmax
            ProbVect=zeros(dmax,1);
            ProbVect(unique(Cnew_aa))=likelihood_fn_str(Y_aa(tt,:),Parameters,...
                likelihood_type,beta*KK_level(aa),mu_r_K(aa,:),unique(Cnew_aa));
            if sum(ProbVect)>0
                ProbVect=ProbVect./sum(ProbVect);
                Cprop_aa(tt)=randsample(dmax,1,'true',ProbVect);
            end
        end
        
        % Cnew
        likelihood=likelihood_fn_str(Ynew_aa,Parameters,likelihood_type,...
            beta*KK_level(aa),mu_r_K(aa,:));
        
        ProbVect = ones(Tnew(aa),dmax);
        for kk=1:dmax
            ProbVect(:,kk)=lambda([kk*ones(Tnew(aa),1),z00_aa]) .* likelihood(:,kk);
        end
        II = sum(ProbVect,2)>0;
        ProbVect = ProbVect./sum(ProbVect,2);
        for tt=1:Tnew(aa)
            if II(tt)
                Cnewprop_aa(tt) = randsample(1:dmax,1,'true',ProbVect(tt,:));
            end
        end
        Cprop_aa((qmax+1):T_K(aa))=Cnewprop_aa;
        for j=1:qmax
             Xnewprop_aa(:,j)=Cprop_aa((qmax+1-j):(T_K(aa)-j));
        end
        % z given C according to G
        for j=1:qmax
            Z_K_prop_aa(:,j)=G(j,Xnewprop_aa(:,j));
        end
        z00prop_aa=Z_K_prop_aa(:,ind00);
        
        logaccprob = log(lambda([Cnewprop_aa,z00prop_aa]))...
                -log(lambda([Cnew_aa,z00_aa]))+log(lambda([Cnew_aa,z00prop_aa]))...
                -log(lambda([Cnewprop_aa,z00_aa]));
        logaccprob = sum(logaccprob);
        
        temperature=max(temperature0^(1-k/2000),1);
        accprob=(exp(logaccprob))^(1/temperature);    % Simulated annealing
        sumk=sum(sum(diff(sort(Z_K_prop_aa))~=0)+1);
        if (rand<accprob) && sumk>qmax+1
            CC_K{aa}=Cprop_aa;
            Z_K{aa}=Z_K_prop_aa;
            Cnew{aa}=Cnewprop_aa;
            Xnew{aa}=Xnewprop_aa;
        end
        L00_K(aa)=length(unique(Cnew{aa}));
    end
    
    L00=max(L00_K);
    
    
    % Update Emission Parameters
    MuY=Parameters{1};
    SigmaY=Parameters{2};
    tbl=zeros(dmax,3); 
    C_total=zeros(sum(T_K),1); % total states
    for aa=1:KK
        C_total((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)))=CC_K{aa};
    end
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
            mu_r_all=zeros(sum(T_K),ObsDimension); 
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
            mu_fix_all=zeros(sum(T_K),ObsDimension);
            for aa=1:KK
                mu_fix_all((sum(T_K(1:(aa-1)))+1):sum(T_K(1:aa)),:)=repmat(beta*KK_level(aa),T_K(aa),1);
            end
        end
      
    end
    
    
    % z
    for ll=1:level_N
        M00=M00_L{ll}; % initiate M00={k_{1},...,k_{p}}, current values of k_{j}'s for the kth iteration
        G=G_L{ll};
        Mact=Mact_L{ll};
        idx_level=find(KK_level==ll);
        ZZ_K_level=cell(1,length(idx_level));
        Z_K_level=cell(1,length(idx_level));
        Cnew_level=cell(1,length(idx_level));
        
    for j=1:qmax
        M0=M00(j);  % M0=k_{j}, current value
        if (M0==1)    % if x_{j} is not included, propose its inclusion OR a switch with an existing important predictor
        	% propose the inclusion of x_{j} with randomly generated cluster mappings for different levels of x_{j}
            new=binornd(1,0.5*ones(1,dmax-1));	% propose new mapping for (d_{j}-1) values at 1
            while sum(new)==0
                new=binornd(1,0.5*ones(1,dmax-1));
            end
            GG=G(j,1:dmax)+[0 new]; % keep the first one at 1, propose new cluster mappings for the other (d_{j}-1) levels of x_{j}
            
            ZZ_K=Z_K;
            for aa=1:length(idx_level)
                zz_aa=Z_K{idx_level(aa)};
                Xnew_aa=Xnew{idx_level(aa)};
                zz_aa(:,j)=GG(Xnew_aa(:,j)); 
                ZZ_K{idx_level(aa)}=zz_aa;
                ZZ_K_level{aa}=zz_aa;
                Cnew_level{aa}=Cnew{idx_level(aa)};
                Z_K_level{aa}=Z_K{idx_level(aa)};
            end
           
            ind2=find(M00>1);
            if isempty(ind2)        % if no predictor is currently important
               ind2=1;
            end
            MM=M00; 
            MM(j)=2;
            ind1=find(MM>1);
            
            logR=logml_multiple_sequence(ZZ_K_level,Cnew_level,MM,pM,alpha_x*lambda_x0(:,ll),dmax,ind1,length(idx_level))...
                -logml_multiple_sequence(Z_K_level,Cnew_level,M00,pM,alpha_x*lambda_x0(:,ll),dmax,ind2,length(idx_level));
            logR=logR+log(0.5)+log(cM(j,dmax)) ; %%%%%% log(0.5) should be cancelled !!!!!!!!!!!!!!
           
            sumk=zeros(length(idx_level),1);
            for aa=1:length(idx_level)
                sumk(aa)=sum(sum(diff(sort(ZZ_K_level{aa}))~=0)+1);
            end
            if log(rand)<logR && min(sumk)>qmax+1
                G(j,1:dmax)=GG;
                M00=MM;
                Z_K=ZZ_K;
            end
        end
        
        if (M0>1)&&(M0<dmax)  % if 1<k_{j}<d_{j} (recall that M0=M00(j)=k_{j})
            if (rand<0.5)     % with prob 0.5 split one mapped value into two
                [~,mm]=unique(sort(G(j,:)),'legacy'); % z0 are unique cluster mappings, mm contains their positions
                gn=mm-[0 mm(1:(end-1))];            % frequencies of z0
                pgn=cM(j,gn)/sum(cM(j,gn));         % see the definition of cM
                rr=sum(mnrnd(1,pgn).*(1:M0));       % rr is the state to split, gn(rr) is the frequency of rr
                new=binornd(1,0.5*ones(1,gn(rr)-1));% propose new mapping for (gn(rr)-1) values at rr
                while sum(new)==0
                    new=binornd(1,0.5*ones(1,gn(rr)-1));
                end
                GG=G(j,1:dmax);
                GG(GG==rr)=rr+(M0+1-rr)*[0 new]; % keep first value at rr, propose new mapping (M0+1) for the rest of (gn(rr)-1) values at rr
                
                ZZ_K=Z_K;
                for aa=1:length(idx_level)
                    zz_aa=Z_K{idx_level(aa)};
                    Xnew_aa=Xnew{idx_level(aa)};
                    zz_aa(:,j)=GG(Xnew_aa(:,j)); 
                    ZZ_K{idx_level(aa)}=zz_aa;
                    ZZ_K_level{aa}=zz_aa;
                    Cnew_level{aa}=Cnew{idx_level(aa)};
                    Z_K_level{aa}=Z_K{idx_level(aa)};
                end
                
                MM=M00;             % MM initizted at current values {k_{1},...,k_{p}}
                MM(j)=M0+1;         % proposed new value of k_{j}, since one original mapped value is split into two
                ind1=find(MM>1);    % proposed set of important predictors
                ind2=find(M00>1);   % current set of important predictors
                if isempty(ind2)
                    ind2=1;
                end                
                logR=logml_multiple_sequence(ZZ_K_level,Cnew_level,MM,pM,alpha_x*lambda_x0(:,ll),dmax,ind1,length(idx_level))...
                    -logml_multiple_sequence(Z_K_level,Cnew_level,M00,pM,alpha_x*lambda_x0(:,ll),dmax,ind2,length(idx_level));
                if M00(j)<dmax-1
                    logR=logR-log(M0*(M0+1)/2)+log(sum(cM(j,gn)));
                else
                    logR=logR-log(dmax*(dmax-1)/2)-log(0.5);%%%%%% log(0.5) should be cancelled !!!!!!!!!!!!!!
                end
                sumk=zeros(length(idx_level),1);
                for aa=1:length(idx_level)
                    sumk(aa)=sum(sum(diff(sort(ZZ_K_level{aa}))~=0)+1);
                end
                if log(rand)<logR && min(sumk)>qmax+1
                    G(j,1:dmax)=GG;
                    M00=MM;
                    Z_K=ZZ_K;
                end
            else        % with prob 0.5 merge two mapped values into one 
                cnew=randsample(M0,2);
                lnew=max(cnew);
                snew=min(cnew);
                GG=G(j,1:dmax);
                GG(GG==lnew)=snew;  % replace all lnews by snews
                GG(GG==M0)=lnew;    % replace the largest cluster mapping by lnew (lnew itself may equal M0, in which case GG remains unchanged by this move)
                
                ZZ_K=Z_K;
                for aa=1:length(idx_level)
                    zz_aa=Z_K{idx_level(aa)};
                    Xnew_aa=Xnew{idx_level(aa)};
                    zz_aa(:,j)=GG(Xnew_aa(:,j)); 
                    ZZ_K{idx_level(aa)}=zz_aa;
                    ZZ_K_level{aa}=zz_aa;
                    Cnew_level{aa}=Cnew{idx_level(aa)};
                    Z_K_level{aa}=Z_K{idx_level(aa)};
                end
                
                MM=M00;             % MM initizted at current values {k_{1},...,k_{p}}, with k_{j}=d_{j} by the if condition
                MM(j)=M00(j)-1;     % proposed new value of k_{j}, since two mappings are merged
                ind1=find(MM>1);    % proposed set of important predictors, may not include x_{j} if original k_(j) was at 2
                ind2=find(M00>1);   % current set of important predictors
                if isempty(ind1)
                    ind1=1;
                end
                if isempty(ind2)
                    ind2=1;
                end
                logR=logml_multiple_sequence(ZZ_K_level,Cnew_level,MM,pM,alpha_x*lambda_x0(:,ll),dmax,ind1,length(idx_level))...
                    -logml_multiple_sequence(Z_K_level,Cnew_level,M00,pM,alpha_x*lambda_x0(:,ll),dmax,ind2,length(idx_level));
                if M0>2
                    [~,mm]=unique(sort(GG),'legacy');
                    gn=mm-[0 mm(1:(end-1))];
                    logR=logR-log(sum(cM(j,gn)))+log(M00(j)*(M00(j)-1)/2); %%% M00(j) -> M0
                else
                    logR=logR-log(cM(j,dmax))-log(0.5); %%%%%% log(0.5) should be cancelled !!!!!!!!!!!!!!
                end
                
                sumk=zeros(length(idx_level),1);
                for aa=1:length(idx_level)
                    sumk(aa)=sum(sum(diff(sort(ZZ_K_level{aa}))~=0)+1);
                end
                if log(rand)<logR && min(sumk)>qmax+1
                    G(j,1:dmax)=GG;
                    M00=MM;
                    Z_K=ZZ_K;
                end
            end
        end        
        if (M0==dmax) % if M0=k_{j}=d_{j}, propose a merger of two different cluster mappings
            cnew=randsample(dmax,2);
            lnew=max(cnew);
            snew=min(cnew);
            GG=G(j,1:dmax);
            GG(GG==lnew)=snew;      % replace all lnews by snews
            GG(GG==M0)=lnew;        % replace the largest cluster mapping d_{j} by lnew (lnew itself can be d_{j}, in which case GG remains unchanged by this move)
            
            ZZ_K=Z_K;
            for aa=1:length(idx_level)
                zz_aa=Z_K{idx_level(aa)};
                Xnew_aa=Xnew{idx_level(aa)};
                zz_aa(:,j)=GG(Xnew_aa(:,j)); 
                ZZ_K{idx_level(aa)}=zz_aa;
                ZZ_K_level{aa}=zz_aa;
                Cnew_level{aa}=Cnew{idx_level(aa)};
                Z_K_level{aa}=Z_K{idx_level(aa)};
            end
                
            MM=M00;                 % MM initizted at current values {k_{1},...,k_{p}}, with k_{j}=d_{j} by the if condition
            MM(j)=dmax-1;           % proposed new value of k_{j}, since originally k_{j}=d_{j} and now two mappings are merged
            ind1=find(MM>1);        % proposed set of important predictors, does not include x_{j} when d_(j)=2
            if isempty(ind1)
               ind1=1;
            end
            ind2=find(M00>1);       % current set of important predictors
            if isempty(ind2)
               ind2=1;
            end            
            logR=logml_multiple_sequence(ZZ_K_level,Cnew_level,MM,pM,alpha_x*lambda_x0(:,ll),dmax,ind1,length(idx_level))...
                 -logml_multiple_sequence(Z_K_level,Cnew_level,M00,pM,alpha_x*lambda_x0(:,ll),dmax,ind2,length(idx_level));
            logR=logR+log(0.5)+log(dmax*(dmax-1)/2); %%%%%% log(0.5) should be cancelled !!!!!!!!!!!!!!
           
            sumk=zeros(length(idx_level),1);
            for aa=1:length(idx_level)
                sumk(aa)=sum(sum(diff(sort(ZZ_K_level{aa}))~=0)+1);
            end
            if log(rand)<logR && min(sumk)>qmax+1
               G(j,1:dmax)=GG;
               M00=MM;
               Z_K=ZZ_K;
            end
        end
        
        % permute the cluster mappings for different lavels of y_{t-j}, e.g. with dmax=4 per={3,4,2,1} maps cluster mappings {1,2,1,1} to {1,1,2,1}    
        if M00(j)>1                     % when M0=dmax, the likelihood is invariariant to permutation of the levels, so the moves are always accepted   
            per=randsample(dmax,dmax);
            GG=G(j,per);            % proposed new cluster mappings of different levels of y_{t-j}
            ZZ_K=Z_K;
            for aa=1:length(idx_level)
                zz_aa=Z_K{idx_level(aa)};
                Xnew_aa=Xnew{idx_level(aa)};
                zz_aa(:,j)=GG(Xnew_aa(:,j)); 
                ZZ_K{idx_level(aa)}=zz_aa;
                ZZ_K_level{aa}=zz_aa;
                Cnew_level{aa}=Cnew{idx_level(aa)};
                Z_K_level{aa}=Z_K{idx_level(aa)};
            end           
            ind1=find(M00>1);
            ind2=find(M00>1);
            logR=logml_multiple_sequence(ZZ_K_level,Cnew_level,M00,pM,alpha_x*lambda_x0(:,ll),dmax,ind1,length(idx_level))...
                 -logml_multiple_sequence(Z_K_level,Cnew_level,M00,pM,alpha_x*lambda_x0(:,ll),dmax,ind2,length(idx_level));
     
            sumk=zeros(length(idx_level),1);
            for aa=1:length(idx_level)
                sumk(aa)=sum(sum(diff(sort(ZZ_K_level{aa}))~=0)+1);
            end
            if log(rand)<logR && min(sumk)>qmax+1
               G(j,1:dmax)=GG;
               Z_K=ZZ_K;
            end
        end
        % if k_{j} remains at one, propose a switch of y_{t-j} with a randomly selected predictor from the current set of important predictors 
        if (M00(j)==1) && (k<N1/2)  % if the y_{t-j} is still not included but some other predictors are included
            ind2=find(M00>1);       % the current set of important predictors
            tempind=randsample(length(ind2),1);  
            temp=ind2(tempind);     % choose one, y_{t-temp}, from the current set of important predictors
            per=randsample(dmax,dmax);
            GG=G(temp,per);
            
            ZZ_K=Z_K;
            for aa=1:length(idx_level)
                zz_aa=Z_K{idx_level(aa)};
                Xnew_aa=Xnew{idx_level(aa)};
                zz_aa(:,temp)=ones(Tnew(idx_level(aa)),1);
                zz_aa(:,j)=GG(Xnew_aa(:,j)); 
                ZZ_K{idx_level(aa)}=zz_aa;
                ZZ_K_level{aa}=zz_aa;
                Cnew_level{aa}=Cnew{idx_level(aa)};
                Z_K_level{aa}=Z_K{idx_level(aa)};
            end 
           
            MM=M00;                 % MM initiated at current values {k_{1},...,k_{p}}, with k_{j}=1 and k_{temp}>1 by the conditions
            MM(temp)=1;             % proposed new value of k_{temp}, since y_{t-temp} is removed from the set of important predictors
            MM(j)=M00(temp);        % propose k_{j}=k_{temp}
            ind1=find(MM>1);        % proposed set of important predictors, now this set excludes y_{t-temp} but includes y_{t-j} 
            
            logR=logml_multiple_sequence(ZZ_K_level,Cnew_level,MM,pM,alpha_x*lambda_x0(:,ll),dmax,ind1,length(idx_level))...
                 -logml_multiple_sequence(Z_K_level,Cnew_level,M00,pM,alpha_x*lambda_x0(:,ll),dmax,ind2,length(idx_level));
       
            sumk=zeros(length(idx_level),1);
            for aa=1:length(idx_level)
                sumk(aa)=sum(sum(diff(sort(ZZ_K_level{aa}))~=0)+1);
            end
            if log(rand)<logR && min(sumk)>qmax+1
               G(j,1:dmax)=GG;
               G(temp,:)=ones(1,dmax);
               M00=MM;
               Z_K=ZZ_K;
            end
        end
    end
    AA=zeros(length(idx_level),qmax);
    for aa=1:length(idx_level)
        AA(aa,:)=sum(diff(sort(Z_K{idx_level(aa)}))~=0)+1;
    end
    Mact(k,:)=max(AA);
    ind0=find(Mact(k,:)==1);
    for aa=1:length(idx_level)
        z_aa=Z_K{idx_level(aa)};
        z_aa(:,ind0)=1;
        Z_K{idx_level(aa)}=z_aa;
        Z_K_level{aa}=z_aa;
    end
    G(ind0,:)=1;
    M00(:,ind0)=1;
    M(k+1,:)=M00;
    M_L{ll}=M;
    G_L{ll}=G;
    Mact_L{ll}=Mact;
    % print informations in each iteration
    npact=length(find(Mact(k,:)>1));
    ind1=find(M00>1);
    if isempty(ind1)
        ind1=1;
    end   
    log0(k)=logml_multiple_sequence(Z_K_level,Cnew_level,M(k+1,:),pM,alpha_x*lambda_x0(:,ll),dmax,ind1,length(idx_level));
    if isnan(log0(k))
        break;
    end
    [~,bC]=find(Mact(k,1:qmax)-1); % find Mact!=1
    fprintf('k = %i, states = [%i], %i predictors = {',k,L00,npact);
    for i=1:length(bC)
        fprintf(' C(t-%i)(%i)',bC(i),Mact(k,bC(i)));
    end
    fprintf(' }. %f \n', log0(k));
    
    end
    L(k+1,:)=L00; % # of states
end

out1=Mact_L;
out2=C_total;
out3=CC_K;
out4=Xnew;
out5=Cnew;
out6=Parameters;
out7=mu_r_K;
out8=beta;
out9=lambda_x;
out10=lambda_x0;
out11=lambda00;
end

