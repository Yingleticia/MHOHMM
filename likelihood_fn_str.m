function [y] = likelihood_fn_str(x,parameters,type,beta,mu_rand,component)

if nargin==5 % number of inputs
    sz=size(parameters{1});
    component=1:sz(1);
end
m=length(component);% number of states

if strcmp(type,'Normal')
    n=length(x); % number of obserations
    mu_temp=parameters{1};
    mu=mu_temp(component)+mu_rand+beta;
    sigma_temp=parameters{2};
    sigma=sqrt(sigma_temp(component));
    zzz=(repmat(x,1,m)-repmat(mu,1,n)')./repmat(sigma,1,n)';
    y=exp(-0.5*(zzz.^2))./(sqrt(2*3.1415926536).*repmat(sigma,1,n)');% n*m matrix

elseif strcmp(type,'MVN')
    n=size(x,1); % number of obserations
    mu_temp=parameters{1};
    Mu=mu_temp(component,:)+mu_rand+beta;
    sigma_temp=parameters{2};
    Sigma=sigma_temp(:,:,component);
    y=zeros(n,m);
    for jj=1:m
        y(:,jj)=mvnpdf(x,Mu(jj,:),Sigma(:,:,jj));
    end
end
end
