function loglik=logml_multiple_sequence(z,Y,M,pM,alpha,d0,ind,KK)

M_=M(ind);
pM_=pM(ind,:);

Y_aa=Y{1};
z_aa=z{1};
z_aa=z_aa(:,ind);
[z0,m]=unique(sortrows([Y_aa z_aa]),'rows','legacy');
C=tensor(zeros([d0 M_]),[d0 M_]);
C(z0)=C(z0)+m-[0;m(1:(end-1))];
Cdata=tenmat(C,1);
Cdata_total=Cdata;

for aa=2:KK
    Y_aa=Y{aa};
    z_aa=z{aa};
    z_aa=z_aa(:,ind);
    [z0,m]=unique(sortrows([Y_aa z_aa]),'rows','legacy');
    C=tensor(zeros([d0 M_]),[d0 M_]);
    C(z0)=C(z0)+m-[0;m(1:(end-1))];
    Cdata=tenmat(C,1);
    Cdata_total=Cdata_total+Cdata;
end

JJ=size(Cdata_total);
if(length(alpha)==1)
    loglik=sum(sum(gammaln(Cdata_total.data+alpha)))-sum(gammaln(sum(Cdata_total.data,1)+d0*alpha))...
        -JJ(2)*(d0*gammaln(alpha)-gammaln(d0*alpha));
else
    loglik=sum(sum(gammaln(Cdata_total.data+repmat(alpha,1,JJ(2)))))-sum(gammaln(sum(Cdata_total.data,1)...
        +repmat(sum(alpha),1,JJ(2))))-JJ(2)*(sum(gammaln(alpha))-gammaln(sum(alpha)));
end

p=size(pM_,1);
for j=1:p
    d=sum(pM_(j,:)>0);
    loglik=loglik+log(pM_(j,M_(j))/stirling(d,M_(j))); % I think d is length(unique(Y))!!!!!!!!!!!
end

