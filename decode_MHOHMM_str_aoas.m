function [LogLikeli,CC,para_idx]=decode_MHOHMM_str_aoas(data,group,lambda_x_storage,...
    lambda_x0_storage,alpha_x0_storage,omega_storage,pi_storage,mu_storage,...
    sigma_storage,beta_storage,Sigma_r_storage,likelihood_type,qmax,dmax,...
    sampsize,p00_L,K00_L,ind00_L,num_para,model_str)

% Model structure
% 1 -- fixed effect in hidden process;
% 2 -- random effect in hidden process;
% 3 -- fixed effect in observed process; 
% 4 -- random effect in observed process;
% 5 -- higher order in the hidden process.

if model_str(1)==0 % no fixed effects in hidden process
    group=1;
end

if model_str(5)==0 % first order
    qmax=1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sz=size(alpha_x0_storage,2); % number of storage
para_idx=randsample(1:sz,num_para); % last 20 posterior samples for model parameters
T=size(data,1);
Tnew=T-qmax;
Y=data;
CC=zeros(T,num_para);
LogLikeli=zeros(1,num_para);

for aa=1:num_para
    idx_aa=para_idx(aa);
    
    % load lambda
    lambda_x0=lambda_x0_storage(:,group,idx_aa);
    lambda_x=lambda_x_storage{group,idx_aa};
    omega=omega_storage(group,:,idx_aa); % weights
    alpha_x0=alpha_x0_storage(group,idx_aa);
    lambda_aa=lambda_x; % initialize
    
    % initialize psi    
    psi=binornd(1,omega(2),[Tnew,1]);
    
    if model_str(2)==0 % no random effect in hidden process
        % omega
        omega=[0,1];
        % psi
        psi=binornd(1,1,[Tnew,1]);
    end

    % load pi
    pi=pi_storage{group,idx_aa};
    ind00=ind00_L{group};
    p00=p00_L{group};
    K00=K00_L{group};
    
    % mu & sigma & mu_r & beta
    if strcmp(likelihood_type,'Normal')
          mu=mu_storage(:,idx_aa);
          sigma=sigma_storage(:,idx_aa);
          beta=beta_storage(idx_aa);
          if model_str(4)==0
              mu_r=0;
          elseif model_str(4)==1
              Sigma_r=Sigma_r_storage(idx_aa);
              mu_r=normrnd(0,sqrt(Sigma_r),1);
          end
          if model_str(3)==0
              beta=0;
          end
    elseif strcmp(likelihood_type,'MVN')
          ObsDimension=size(Y,2);
          mu=mu_storage(:,:,idx_aa); 
          sigma=sigma_storage(:,:,:,idx_aa);
          beta=beta_storage(:,:,idx_aa);
          if model_str(4)==0
              mu_r=zeros(1,ObsDimension);
          elseif model_str(4)==1
              Sigma_r=Sigma_r_storage{idx_aa};
              mu_r=mvnrnd(zeros(1,ObsDimension),Sigma_r,1); 
          end
          if model_str(3)==0
              beta=zeros(1,ObsDimension);
          end
    end
    Parameters=cell(1,2);
    Parameters{1}=mu;
    Parameters{2}=sigma;
    
    % initialize c
    C=zeros(T,1);
    for ee=1:T
        [~,idx] = pdist2(mu+beta*group,Y(ee,:),'euclidean','Smallest',1);
        C(ee)=idx; % initial states
    end
    Cnew=C((qmax+1):end);
    Xnew=C(((qmax+1):T)'-(1:qmax));
    x00=Xnew(:,ind00);
    
    % initialize z
    z00=ones(Tnew,p00);
    for cc=1:p00
        prob_z=zeros(Tnew,K00(cc));
        for dd=1:K00(cc)
            prob_z(:,dd)=pi(cc,x00(:,cc),dd);
        end
        prob_z(prob_z==0)=10^(-5);%%%% in case NAN
        prob_z=bsxfun(@rdivide,prob_z,(sum(prob_z,2)));
        z00(:,cc)=sum(bsxfun(@times,mnrnd(1,prob_z),1:K00(cc)),2);
    end
   
    
    % for storage
    C_storage=zeros(T,sampsize);
    LogLikeli_storage=zeros(1,sampsize);
    
%%
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% iteration start %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%
   for ii=1:sampsize
       
       if model_str(2)==1 % random effect in hidden process
           
            %%%%%% psi %%%%%%
            prob = omega.*[lambda_aa([Cnew,z00]) lambda_x([Cnew,z00])];
            prob(prob==0)=10^(-5);%%%% in case NAN
            prob=bsxfun(@rdivide,prob,sum(prob,2));
            psi=binornd(ones(1,Tnew),prob(:,2)')';
            
            %%%%%% lambda_aa %%%%%%
            if sum(psi==0)~=0
                z00_temp=z00(psi==0,:); % psi=0: random term
                Cnew_temp=Cnew(psi==0); % psi=0: random term
                clT=tensor(zeros([dmax,K00]),[dmax,K00]);
                [z0,m]=unique(sortrows([Cnew_temp z00_temp]),'rows','legacy');
                clT(z0)=clT(z0)+m-[0;m(1:(end-1))];
                clTdata=tenmat(clT,1);
                sz=size(clTdata);
                n_aa=clTdata.data;
                lambdamat=zeros(dmax,sz(2));
                for j=1:sz(2)
                   lambdamat(:,j)=gamrnd(n_aa(:,j)+alpha_x0*lambda_x0,1);
                   while sum(lambdamat(:,j))==0
                         lambdamat(:,j)=gamrnd(n_aa(:,j)+alpha_x0*lambda_x0,1);
                   end
                   lambdamat(:,j)=lambdamat(:,j)/sum(lambdamat(:,j));
                end
                lambda_aa=tensor(lambdamat,[dmax,K00]);
            end
            
       end
       
       
       %%%%%% z %%%%%%
       prob2_hh = cell(1,p00);
       iii = ones(Tnew,1);
       z00_ori = z00;
       if model_str(2)==0 % no random effect in hidden process
           for j=1:p00
               prob_hhh = zeros(Tnew,K00(j));
               for h=1:K00(j)
                   II = [Cnew,z00(:,1:(j-1)),h*iii,z00_ori(:,(j+1):p00)];
                   prob_hhh(:,h)=pi(j,x00(:,j),h)'.*lambda_x(II);
               end
           end
           prob_hhh(prob_hhh==0)=10^(-5);%%%% in case NAN
           prob2_hh{j} = prob_hhh./sum(prob_hhh,2);
           z00(:,j)=sum(mnrnd(1,prob2_hh{j}).*(1:K00(j)),2);
       elseif model_str(2)==1 % random effect in hidden process
           JJ = psi~=0; % fixed effect
           for j=1:p00
               prob_hh = zeros(Tnew,K00(j));
               prob_hhh = zeros(Tnew,K00(j));
               for h=1:K00(j)
                   II = [Cnew,z00(:,1:(j-1)),h*iii,z00_ori(:,(j+1):p00)];
                   prob_hh(:,h)=pi(j,x00(:,j),h)'.*lambda_aa(II);
                   prob_hhh(:,h)=pi(j,x00(:,j),h)'.*lambda_x(II);
               end
               prob_hh(prob_hh==0)=10^(-5);%%%% in case NAN
               prob_hhh(prob_hhh==0)=10^(-5);%%%% in case NAN
               prob2_hh{j} = prob_hh./sum(prob_hh,2);
               prob2_hh{j}(JJ,:) = prob_hhh(JJ,:)./sum(prob_hhh(JJ,:),2);
               z00(:,j)=sum(mnrnd(1,prob2_hh{j}).*(1:K00(j)),2);
           end
       end    
       
       
       %%%%%% C %%%%%%
       for tt=1:qmax
           ProbVect=zeros(dmax,1);
           ProbVect(unique(Cnew))=likelihood_fn_str(Y(tt,:),Parameters,...
               likelihood_type,beta*group,mu_r,unique(Cnew));
           if sum(ProbVect)>0
              ProbVect=ProbVect./sum(ProbVect);
              C(tt)=randsample(dmax,1,'true',ProbVect);
           end
       end
       % Cnew
       likelihood=likelihood_fn_str(Y((qmax+1):end,:),Parameters,...
           likelihood_type,beta*group,mu_r);
       ProbVect=ones(Tnew,dmax);
       for kk=1:dmax
           if model_str(2)==0 % no random effect in hidden process
               lambda_sub = lambda_x([kk*ones(Tnew,1),z00]); 
           elseif model_str(2)==1 % random effect in hidden process
                lambda_sub = lambda_aa([kk*ones(Tnew,1),z00]);
                lambda_fix_sub = lambda_x([kk*ones(Tnew,1),z00]);
                lambda_sub(psi~=0,:) = lambda_fix_sub(psi~=0,:);
           end
           II_z00 = (1:(Tnew-qmax))'+ind00;
           hhh = z00(II_z00',:);
           [nrh, nch] = size(hhh);
           iii = sub2ind([nrh, nch],(1:nrh)',repmat((1:nch)',nrh/nch,1));
           hhh = hhh(iii);
           pi_sub = ones(Tnew,1);
           temp = reshape(pi(:,kk,hhh),p00,[]);
           temp = temp';
           temp = prod(reshape(temp(iii),p00,[]),1);
           pi_sub(1:(Tnew-qmax),:) = temp';
           likelihood_sub = likelihood(:,kk);
           ProbVect(:,kk) = lambda_sub.*pi_sub.*likelihood_sub;
       end
       II = sum(ProbVect,2)>0;
       ProbVect = ProbVect./sum(ProbVect,2);
       for tt=1:Tnew
           if II(tt)
               Cnew(tt) = randsample(1:dmax,1,'true',ProbVect(tt,:));
           end
       end
       C((qmax+1):T)=Cnew;
       Xnew=C(((qmax+1):T)'-(1:qmax));
       x00=Xnew(:,ind00);
       
       %%%%%% mu_r %%%%%%
       if model_str(4)==1 
           tbl_aa=zeros(dmax,3);
           tbl_aa(1:max(C),:)=tabulate(C);
           
           if strcmp(likelihood_type,'Normal')
               Mu_r_SigmaSq=1/(1/Sigma_r+sum(tbl_aa(:,2)./sigma));
               temp=sum((Y-mu(C)-beta*group)./sigma(C));
               Mu_r_Mu=Mu_r_SigmaSq*temp;
               mu_r=normrnd(Mu_r_Mu,sqrt(Mu_r_SigmaSq),1);
           elseif strcmp(likelihood_type,'MVN')
               temp_Sigma=inv(Sigma_r);
               temp_Mu=zeros(size(mu));
               for mm=1:dmax
                    if tbl_aa(mm,2)>1
                        temp_Sigma=temp_Sigma+inv(sigma(:,:,mm)/tbl_aa(mm,2));
                        temp_Mu(mm,:)=(sigma(:,:,mm)\((sum(Y(C==mm,:)...
                            -mu(mm,:)-beta*group))'))';
                    elseif tbl_aa(mm,2)==1
                        temp_Sigma=temp_Sigma+inv(sigma(:,:,mm)/tbl_aa(mm,2));
                        temp_Mu(mm,:)=(sigma(:,:,mm)\((Y(C==mm,:)-...
                            mu(mm,:)-beta*group)'))';
                    end
               end
                Mu_r=temp_Sigma\((sum(temp_Mu))');
                mu_r=mvnrnd(Mu_r',inv(temp_Sigma),1);
            end
       end
      
       % storage
       C_storage(:,ii)=C;  
       if strcmp(likelihood_type,'Normal')
           LogLikeli_storage(ii)=sum(log(normpdf(Y,mu(C)+mu_r+beta*group,sqrt(sigma(C)))));
       elseif strcmp(likelihood_type,'MVN')
           LogLikeli_storage(ii)=sum(log(mvnpdf(Y,mu(C,:)+mu_r+beta*group,sigma(:,:,C))));
       end
       
   end
   
   C_mode=mode(C_storage(:,(sampsize/2+1):sampsize),2); 
   CC(:,aa)=C_mode;
   LogLikeli(aa)=mean(LogLikeli_storage((sampsize/2+1):sampsize));
%    LogLikeli(aa)=max(LogLikeli_storage((sampsize/2+1):sampsize));
end


end