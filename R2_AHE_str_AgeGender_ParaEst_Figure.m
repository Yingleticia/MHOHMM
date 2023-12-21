

clear;clc;close all;

%% Setting
seed = 46; % best accuracy: 46
% (good ones: 4; 6)

seed_opt = 11;


gap=1; % 1: no gap; 2: 30-min gap
model=1; % 1: MHOHMM 2: HOHMM 3: MHMM

fig_now=1;

%% Posterior samples loading

filename=['Result_AgeGender/Replicates/Seed_',num2str(seed_opt),...
    '/ParaEst_ForFigure_Seed_',num2str(seed),'.mat'];
load(filename) 
% Model
% 'Mact_figure','weight_figure','lambda_x0_figure',
% 'mu_figure','sigma_figure','beta_figure','Sigma_r_figure'
% Gap X class X models



% Group (Age + Gender)
% 1: Young Male
% 2: Young Female
% 3: Elderly Male
% 4: Elderly Female


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Transition orders and k_j's %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% samples loading
Mact_A=Mact_figure{gap,1,model}; % AHE
Mact_NA=Mact_figure{gap,2,model}; % Non-AHE

qmax=5;
N1=500;

disp('========================== q_x & k_j =============================')

%%% AHE %%%
disp('------------------- AHE: Group 1 (Young Male) -------------------')
% group 1
Mact=Mact_A{1};
aveM=mean(Mact(floor(N1/2):N1,:),1);
MM0=round(aveM); 
MProp=(Mact>1);
MProp=mean(MProp(floor(N1/2):N1,:),1);
ind00=find(MProp>0.5); 
K00=MM0(ind00); 
MactFirstStage=Mact;
MactFirstStage=MactFirstStage(floor(N1/2):N1,:);
MProp=zeros(size(MactFirstStage));
for ii=1:qmax
    MProp(MactFirstStage(:,ii)>1,ii)=ii;
end
temp11=max(MProp,[],2);
order_mode = mode(temp11);
order_freq = 100*sum(temp11==order_mode)/length(temp11);
disp(['==> q_1 = ',num2str(order_mode),' with frequency ',num2str(order_freq),'%'])
for i=1:length(ind00)
    jj = ind00(i);
    k_j = K00(jj);
    prob_k_j = 100*mean(MactFirstStage(:,jj)==k_j);
    disp(['     --> k_',num2str(jj),' = ',num2str(k_j),' with frequency ',num2str(prob_k_j),'%'])
end

% group 2
disp('------------------- AHE: Group 2 (Young Female) -------------------')
Mact=Mact_A{2};
aveM=mean(Mact(floor(N1/2):N1,:),1);
MM0=round(aveM); 
MProp=(Mact>1);
MProp=mean(MProp(floor(N1/2):N1,:),1);
ind00=find(MProp>0.5); 
K00=MM0(ind00); 
MactFirstStage=Mact;
MactFirstStage=MactFirstStage(floor(N1/2):N1,:);
MProp=zeros(size(MactFirstStage));
for ii=1:qmax
    MProp(MactFirstStage(:,ii)>1,ii)=ii;
end
temp12=max(MProp,[],2);
order_mode = mode(temp12);
order_freq = 100*sum(temp12==order_mode)/length(temp12);
disp(['==> q_2 = ',num2str(order_mode),' with frequency ',num2str(order_freq),'%'])
for i=1:length(ind00)
    jj = ind00(i);
    k_j = K00(jj);
    prob_k_j = 100*mean(MactFirstStage(:,jj)==k_j);
    disp(['     --> k_',num2str(jj),' = ',num2str(k_j),' with frequency ',num2str(prob_k_j),'%'])
end

% group 3
disp('------------------- AHE: Group 3 (Elderly Male) -------------------')
Mact=Mact_A{3};
aveM=mean(Mact(floor(N1/2):N1,:),1);
MM0=round(aveM); 
MProp=(Mact>1);
MProp=mean(MProp(floor(N1/2):N1,:),1);
ind00=find(MProp>0.5); 
K00=MM0(ind00); 
MactFirstStage=Mact;
MactFirstStage=MactFirstStage(floor(N1/2):N1,:);
MProp=zeros(size(MactFirstStage));
for ii=1:qmax
    MProp(MactFirstStage(:,ii)>1,ii)=ii;
end
temp13=max(MProp,[],2);
order_mode = mode(temp13);
order_freq = 100*sum(temp13==order_mode)/length(temp13);
disp(['==> q_3 = ',num2str(order_mode),' with frequency ',num2str(order_freq),'%'])
for i=1:length(ind00)
    jj = ind00(i);
    k_j = K00(jj);
    prob_k_j = 100*mean(MactFirstStage(:,jj)==k_j);
    disp(['     --> k_',num2str(jj),' = ',num2str(k_j),' with frequency ',num2str(prob_k_j),'%'])
end

% group 4
disp('------------------- AHE: Group 4 (Elderly Female) -------------------')
Mact=Mact_A{4};
aveM=mean(Mact(floor(N1/2):N1,:),1);
MM0=round(aveM); 
MProp=(Mact>1);
MProp=mean(MProp(floor(N1/2):N1,:),1);
ind00=find(MProp>0.5); 
K00=MM0(ind00); 
MactFirstStage=Mact;
MactFirstStage=MactFirstStage(floor(N1/2):N1,:);
MProp=zeros(size(MactFirstStage));
for ii=1:qmax
    MProp(MactFirstStage(:,ii)>1,ii)=ii;
end
temp14=max(MProp,[],2);
order_mode = mode(temp14);
order_freq = 100*sum(temp14==order_mode)/length(temp14);
disp(['==> q_4 = ',num2str(order_mode),' with frequency ',num2str(order_freq),'%'])
for i=1:length(ind00)
    jj = ind00(i);
    k_j = K00(jj);
    prob_k_j = 100*mean(MactFirstStage(:,jj)==k_j);
    disp(['     --> k_',num2str(jj),' = ',num2str(k_j),' with frequency ',num2str(prob_k_j),'%'])
end


%%% Non-AHE %%%
disp('------------------- Non-AHE: Group 1 (Young Male) -------------------')
% group 1
Mact=Mact_NA{1};
aveM=mean(Mact(floor(N1/2):N1,:),1);
MM0=round(aveM); 
MProp=(Mact>1);
MProp=mean(MProp(floor(N1/2):N1,:),1);
ind00=find(MProp>0.5); 
K00=MM0(ind00); 
MactFirstStage=Mact;
MactFirstStage=MactFirstStage(floor(N1/2):N1,:);
MProp=zeros(size(MactFirstStage));
for ii=1:qmax
    MProp(MactFirstStage(:,ii)>1,ii)=ii;
end
temp21=max(MProp,[],2);
order_mode = mode(temp21);
order_freq = 100*sum(temp21==order_mode)/length(temp21);
disp(['==> q_1 = ',num2str(order_mode),' with frequency ',num2str(order_freq),'%'])
for i=1:length(ind00)
    jj = ind00(i);
    k_j = K00(jj);
    prob_k_j = 100*mean(MactFirstStage(:,jj)==k_j);
    disp(['     --> k_',num2str(jj),' = ',num2str(k_j),' with frequency ',num2str(prob_k_j),'%'])
end

disp('------------------- AHE: Group 2 (Young Female) -------------------')
% group 2
Mact=Mact_NA{2};
aveM=mean(Mact(floor(N1/2):N1,:),1);
MM0=round(aveM); 
MProp=(Mact>1);
MProp=mean(MProp(floor(N1/2):N1,:),1);
ind00=find(MProp>0.5); 
K00=MM0(ind00); 
MactFirstStage=Mact;
MactFirstStage=MactFirstStage(floor(N1/2):N1,:);
MProp=zeros(size(MactFirstStage));
for ii=1:qmax
    MProp(MactFirstStage(:,ii)>1,ii)=ii;
end
temp22=max(MProp,[],2);
order_mode = mode(temp22);
order_freq = 100*sum(temp22==order_mode)/length(temp22);
disp(['==> q_2 = ',num2str(order_mode),' with frequency ',num2str(order_freq),'%'])
for i=1:length(ind00)
    jj = ind00(i);
    k_j = K00(jj);
    prob_k_j = 100*mean(MactFirstStage(:,jj)==k_j);
    disp(['     --> k_',num2str(jj),' = ',num2str(k_j),' with frequency ',num2str(prob_k_j),'%'])
end

disp('------------------- AHE: Group 3 (Elderly Male) -------------------')
% group 3
Mact=Mact_NA{3};
aveM=mean(Mact(floor(N1/2):N1,:),1);
MM0=round(aveM); 
MProp=(Mact>1);
MProp=mean(MProp(floor(N1/2):N1,:),1);
ind00=find(MProp>0.5); 
K00=MM0(ind00); 
MactFirstStage=Mact;
MactFirstStage=MactFirstStage(floor(N1/2):N1,:);
MProp=zeros(size(MactFirstStage));
for ii=1:qmax
    MProp(MactFirstStage(:,ii)>1,ii)=ii;
end
temp23=max(MProp,[],2);
order_mode = mode(temp23);
order_freq = 100*sum(temp23==order_mode)/length(temp23);
disp(['==> q_3 = ',num2str(order_mode),' with frequency ',num2str(order_freq),'%'])
for i=1:length(ind00)
    jj = ind00(i);
    k_j = K00(jj);
    prob_k_j = 100*mean(MactFirstStage(:,jj)==k_j);
    disp(['     --> k_',num2str(jj),' = ',num2str(k_j),' with frequency ',num2str(prob_k_j),'%'])
end

disp('------------------- AHE: Group 4 (Elderly Female) -------------------')
% group 4
Mact=Mact_NA{4};
aveM=mean(Mact(floor(N1/2):N1,:),1);
MM0=round(aveM); 
MProp=(Mact>1);
MProp=mean(MProp(floor(N1/2):N1,:),1);
ind00=find(MProp>0.5); 
K00=MM0(ind00); 
MactFirstStage=Mact;
MactFirstStage=MactFirstStage(floor(N1/2):N1,:);
MProp=zeros(size(MactFirstStage));
for ii=1:qmax
    MProp(MactFirstStage(:,ii)>1,ii)=ii;
end
temp24=max(MProp,[],2);
order_mode = mode(temp24);
order_freq = 100*sum(temp24==order_mode)/length(temp24);
disp(['==> q_4 = ',num2str(order_mode),' with frequency ',num2str(order_freq),'%'])
for i=1:length(ind00)
    jj = ind00(i);
    k_j = K00(jj);
    prob_k_j = 100*mean(MactFirstStage(:,jj)==k_j);
    disp(['     --> k_',num2str(jj),' = ',num2str(k_j),' with frequency ',num2str(prob_k_j),'%'])
end


%%% figure %%%
figure(fig_now);
fig_now=fig_now+1;
tiledlayout(4,2,'TileSpacing','compact'); % loose; compact; tight; none
nexttile(1) 
histogram(temp11,'Normalization','probability','FaceAlpha',0.4); hold on;
line([0 5],[0.5 0.5],'Color','red','LineStyle',':', 'LineWidth', 1.5); 
xlim([0 5])
ylim([0 1])
set(gca,'FontSize',16,'fontname','times');
ylabel('$q_1$ (YM)','Interpreter','latex','FontSize',20)
title('AHE','fontname','times','FontSize',20,'FontWeight','normal')
nexttile(3) 
histogram(temp12,'Normalization','probability','FaceAlpha',0.4);hold on;
line([0 5],[0.5 0.5],'Color','red','LineStyle',':', 'LineWidth', 1.5);
xlim([0 5])
ylim([0 1])
set(gca,'FontSize',16,'fontname','times');
ylabel('$q_2$ (YF)','Interpreter','latex','FontSize',20)
nexttile(5) 
histogram(temp13,'Normalization','probability','FaceAlpha',0.4);hold on;
line([0 5],[0.5 0.5],'Color','red','LineStyle',':', 'LineWidth', 1.5);
xlim([0 5])
ylim([0 1])
set(gca,'FontSize',16,'fontname','times');
ylabel('$q_3$ (EM)','Interpreter','latex','FontSize',20)
nexttile(7) 
histogram(temp14,'Normalization','probability','FaceAlpha',0.4);hold on;
line([0 5],[0.5 0.5],'Color','red','LineStyle',':', 'LineWidth', 1.5);
xlim([0 5])
ylim([0 1])
set(gca,'FontSize',16,'fontname','times');
ylabel('$q_4$ (EF)','Interpreter','latex','FontSize',20)

nexttile(2) 
histogram(temp21,'Normalization','probability','FaceAlpha',0.4);hold on;
line([0 5],[0.5 0.5],'Color','red','LineStyle',':', 'LineWidth', 1.5);
xlim([0 5])
ylim([0 1])
set(gca,'FontSize',16,'fontname','times');
title('Non-AHE','fontname','times','FontSize',20,'FontWeight','normal')
nexttile(4) 
histogram(temp22,'Normalization','probability','FaceAlpha',0.4);hold on;
line([0 5],[0.5 0.5],'Color','red','LineStyle',':', 'LineWidth', 1.5);
xlim([0 5])
ylim([0 1])
set(gca,'FontSize',16,'fontname','times');
nexttile(6) 
histogram(temp23,'Normalization','probability','FaceAlpha',0.4);hold on;
line([0 5],[0.5 0.5],'Color','red','LineStyle',':', 'LineWidth', 1.5);
xlim([0 5])
ylim([0 1])
set(gca,'FontSize',16,'fontname','times');
nexttile(8) 
histogram(temp24,'Normalization','probability','FaceAlpha',0.4);hold on;
line([0 5],[0.5 0.5],'Color','red','LineStyle',':', 'LineWidth', 1.5);
xlim([0 5])
ylim([0 1])
set(gca,'FontSize',16,'fontname','times');

%%
%%%%%%%%%%%%%%%
%%% Weights %%%
%%%%%%%%%%%%%%%
%----- Parameters -----%
ymax=0.3;
xmax=0.07;
FaceAlpha = 0.3;
%----------------------%

disp('========================== Weights (w^0) ==========================')

% samples loading (2 X 2 X 200)
weight_A=weight_figure{gap,1,model}; % AHE
weight_NA=weight_figure{gap,2,model}; % Non-AHE
sz3=size(weight_A,3);

disp('------------------- AHE -------------------')
temp11=weight_A(1,1,:);
temp11=reshape(temp11,[1 sz3]);
weight_mean = mean(temp11);
weight_std = std(temp11);
disp(['--> Group 1 (Young Male): ',num2str(weight_mean),' with std ',num2str(weight_std)])

temp12=weight_A(2,1,:);
temp12=reshape(temp12,[1 sz3]);
weight_mean = mean(temp12);
weight_std = std(temp12);
disp(['--> Group 2 (Young Female): ',num2str(weight_mean),' with std ',num2str(weight_std)])

temp13=weight_A(3,1,:);
temp13=reshape(temp13,[1 sz3]);
weight_mean = mean(temp13);
weight_std = std(temp13);
disp(['--> Group 3 (Elderly Male): ',num2str(weight_mean),' with std ',num2str(weight_std)])

temp14=weight_A(4,1,:);
temp14=reshape(temp14,[1 sz3]);
weight_mean = mean(temp14);
weight_std = std(temp14);
disp(['--> Group 4 (Elderly Female): ',num2str(weight_mean),' with std ',num2str(weight_std)])

disp('------------------- Non-AHE -------------------')
temp21=weight_NA(1,1,:);
temp21=reshape(temp21,[1 sz3]);
weight_mean = mean(temp21);
weight_std = std(temp21);
disp(['--> Group 1 (Young Male): ',num2str(weight_mean),' with std ',num2str(weight_std)])

temp22=weight_NA(2,1,:);
temp22=reshape(temp22,[1 sz3]);
weight_mean = mean(temp22);
weight_std = std(temp22);
disp(['--> Group 2 (Young Female): ',num2str(weight_mean),' with std ',num2str(weight_std)])

temp23=weight_NA(3,1,:);
temp23=reshape(temp23,[1 sz3]);
weight_mean = mean(temp23);
weight_std = std(temp23);
disp(['--> Group 3 (Elderly Male): ',num2str(weight_mean),' with std ',num2str(weight_std)])

temp24=weight_NA(4,1,:);
temp24=reshape(temp24,[1 sz3]);
weight_mean = mean(temp24);
weight_std = std(temp24);
disp(['--> Group 4 (Elderly Female): ',num2str(weight_mean),' with std ',num2str(weight_std)])


%%% figure %%%
color_order = get(gca,'colororder');
BinsN=15;
legend_name={'YM','Mean','YF','Mean',...
    'EM','Mean','EF','Mean'};
figure(fig_now);
fig_now=fig_now+1;
tiledlayout(1,2,'TileSpacing','compact'); % loose; compact; tight; none
nexttile(1) 
h=histogram(temp11,'Normalization','probability','FaceAlpha',FaceAlpha,'FaceColor',color_order(1,:));
h.NumBins = BinsN;
set(gca,'yticklabel',[],'ytick',[],'FontSize',16,'fontname','times')
ylim([0 ymax])
xlim([0 xmax])
title('AHE','fontname','times','FontSize',20,'FontWeight','normal')
hold on;
line([mean(temp11) mean(temp11)],[0 ymax],...
    'linestyle','-.', 'Color',color_order(1,:), 'LineWidth', 1.5);

h=histogram(temp12,'Normalization','probability','FaceAlpha',FaceAlpha,'FaceColor',color_order(2,:));
h.NumBins = BinsN;
line([mean(temp12) mean(temp12)],[0 ymax],...
    'linestyle','-.', 'Color',color_order(2,:), 'LineWidth', 1.5);   

h=histogram(temp13,'Normalization','probability','FaceAlpha',FaceAlpha,'FaceColor',color_order(3,:));
h.NumBins = BinsN;
line([mean(temp13) mean(temp13)],[0 ymax],...
    'linestyle','-.', 'Color',color_order(3,:), 'LineWidth', 1.5);   

h=histogram(temp14,'Normalization','probability','FaceAlpha',FaceAlpha,'FaceColor',color_order(4,:));
h.NumBins = BinsN;
line([mean(temp14) mean(temp14)],[0 ymax],...
    'linestyle','-.', 'Color',color_order(4,:), 'LineWidth', 1.5);   

legend(legend_name,'FontSize',14,'fontname','times') % ,'Location','northwest'
legend('boxoff')

nexttile(2)    
h=histogram(temp21,'Normalization','probability','FaceAlpha',FaceAlpha,'FaceColor',color_order(1,:));
h.NumBins = BinsN;
% set(0,'DefaultAxesTitleFontWeight','normal');
set(gca,'yticklabel',[],'ytick',[],'FontSize',16,'fontname','times')
ylim([0 ymax])
xlim([0 xmax])
title('Non-AHE','fontname','times','FontSize',20,'FontWeight','normal')
hold on;
line([mean(temp21) mean(temp21)],[0 ymax],...
    'linestyle','-.', 'Color',color_order(1,:), 'LineWidth', 1.5);

h=histogram(temp22,'Normalization','probability','FaceAlpha',FaceAlpha,'FaceColor',color_order(2,:));
h.NumBins = BinsN;
line([mean(temp22) mean(temp22)],[0 ymax],...
    'linestyle','-.', 'Color',color_order(2,:), 'LineWidth', 1.5); 

h=histogram(temp23,'Normalization','probability','FaceAlpha',FaceAlpha,'FaceColor',color_order(3,:));
h.NumBins = BinsN;
line([mean(temp23) mean(temp23)],[0 ymax],...
    'linestyle','-.', 'Color',color_order(3,:), 'LineWidth', 1.5); 

h=histogram(temp24,'Normalization','probability','FaceAlpha',FaceAlpha,'FaceColor',color_order(4,:));
h.NumBins = BinsN;
line([mean(temp24) mean(temp24)],[0 ymax],...
    'linestyle','-.', 'Color',color_order(4,:), 'LineWidth', 1.5); 

legend(legend_name,'FontSize',14,'fontname','times') % ,'Location','northwest'
legend('boxoff')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% State prevalence and density functions %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----------- Parameter -----------%
stateNum=4; % 3 or 4
A_state = [1,3,4]; % for figure
NA_state = [1,2,4]; % for figure
ordering=1; % state ordering
%---------------------------------%

% if stateNum==4
%     ordering=0;
% end
CI=1; % CI for density functions
CI_method=2; % 1: t distribution; 2: percentile

%%% AHE %%%
lambda_x0=lambda_x0_figure{gap,1,model};
mu=mu_figure{gap,1,model};
sigma=sigma_figure{gap,1,model};
beta=beta_figure{gap,1,model};
% Sigma_r=Sigma_r_figure{gap,1,model};

% state ordering
temp_mu=mu(1:stateNum,:);
mu_mean=mean(temp_mu,2);
mu_mean_order=sort(mu_mean);
state_order=1:stateNum;
if ordering==1
    for ii=1:stateNum
        state_order(ii)=find(mu_mean==mu_mean_order(ii));
    end
end
disp('=========== State prevalence probabilities (lambda_x^0) ===========')
% State prevalence probabilities  %
sz3=size(lambda_x0,3);
figure(fig_now);
fig_now=fig_now+1;
tiledlayout(2,length(A_state),'TileSpacing','tight'); % loose; compact; tight; none
legend_name={'YM','Mean','YF','Mean',...
    'EM','Mean','EF','Mean'};
ymax=0.2;
BinsN=15;
disp('------------------- AHE -------------------')
prop_all = zeros(4,length(A_state));
for ii=1:length(A_state) 
    nexttile(ii)
    ss = A_state(ii);
    disp(['-----> State ',num2str(ii)])
    for ll=1:4 % level
        temp=lambda_x0(state_order(ss),ll,:);
        temp=reshape(temp,[1 sz3]);
        prob_mean = mean(temp);
        prob_std = std(temp);
        prop_all(ll,ii) = prob_mean;
        disp(['--> Group ',num2str(ll),' : ',num2str(prob_mean),' with std ',num2str(prob_std)])
        h=histogram(temp,'Normalization','probability','FaceAlpha',FaceAlpha,'FaceColor',color_order(ll,:));
        h.NumBins = BinsN;
        ylim([0 ymax])
        xlim([0 1])
        set(gca,'yticklabel',[],'ytick',[],'FontSize',16,'fontname','times')
        name=['State ',num2str(ii)];
        title(name,'FontSize',20,'fontname','times','FontWeight','normal')
        hold on;
        Ylim=ylim;
        line([mean(temp) mean(temp)],[Ylim(1) Ylim(2)],...
            'linestyle','-.', 'Color',color_order(ll,:), 'LineWidth', 1.5);
    end

    if ii==1
        ylabel('$\lambda^0_x$','Interpreter','latex','FontSize',20)
    end
    legend(legend_name,'FontSize',14,'fontname','times') % ,'Location','northwest'
    legend('boxoff')
end
disp('============> Remaining probability ')
for ll=1:4 % level
    disp(['--> Group ',num2str(ll),' : ',num2str(1-sum(prop_all(ll,:)))])
end

% density functions %
xmin=-4;
xmax=6;
ymax=1.2;
yy = linspace(xmin,xmax,100); 
legend_name={'YM','YF','EM','EF'};

for jj=1:length(A_state)
    nexttile(ii+jj)
    ss = A_state(jj);
    for ll=1:4
        temp_mu=mu(state_order(ss),:);
        temp_sigma=sqrt(sigma(state_order(ss),:));
        temp_mu=temp_mu+ll*beta;

        pp = normpdf(yy,temp_mu',temp_sigma');
        p = mean(pp,1);
        %%% CI %%%
        if CI_method==1 % method 1: t-distribution
            p_std = std(pp,1);
            ssz=length(temp_mu);
            p_low = p - tinv(0.975,ssz-1)*(p_std/sqrt(ssz));
            p_up = p + tinv(0.975,ssz-1)*(p_std/sqrt(ssz));
        end
        if CI_method==2 % method 2: percentile
            p_low=prctile(pp,2.5,1);
            p_up=prctile(pp,97.5,1);
        end 
        plot(yy,p,'LineWidth', 2,'Color',color_order(ll,:)); 
        xlim([xmin xmax])
        ylim([0 ymax])
        set(gca,'yticklabel',[],'ytick',[],'FontSize',16,'fontname','times')
        hold on
        if CI==1
            h_temp=fill([yy,fliplr(yy)],[p_low,fliplr(p_up)],color_order(ll,:));
            set(h_temp,'edgealpha',0,'facealpha',0.4)
            legend_name={'YM','95% CI','YF','95% CI',...
                'EM','95% CI','EF','95% CI'};
        end
    end
    if jj==1
        ylabel('Density','FontSize',20)
    end
    legend(legend_name,'FontSize',14,'fontname','times') % ,'Location','northwest'
    legend('boxoff')
end

disp('------------------- Non-AHE -------------------')
% stateNum=3;
prop_all = zeros(4,length(NA_state));
%%% Non-AHE %%%
lambda_x0=lambda_x0_figure{gap,2,model};
mu=mu_figure{gap,2,model};
sigma=sigma_figure{gap,2,model};
beta=beta_figure{gap,2,model};
% Sigma_r=Sigma_r_figure{gap,2,model};

% state ordering
temp_mu=mu(1:stateNum,:);
mu_mean=mean(temp_mu,2);
mu_mean_order=sort(mu_mean);
state_order=1:stateNum;
if ordering==1
    for ii=1:stateNum
        state_order(ii)=find(mu_mean==mu_mean_order(ii));
    end
end

% State prevalence probabilities  %
sz3=size(lambda_x0,3);
figure(fig_now);
fig_now=fig_now+1;
tiledlayout(2,length(A_state),'TileSpacing','tight'); % loose; compact; tight; none
legend_name={'YM','Mean','YF','Mean',...
    'EM','Mean','EF','Mean'};
ymax=0.2;
BinsN=15;
for ii=1:length(NA_state) 
    nexttile(ii)
    ss = NA_state(ii);
    disp(['-----> State ',num2str(ii)])
    for ll=1:4 % level
        temp=lambda_x0(state_order(ss),ll,:);
        temp=reshape(temp,[1 sz3]);
        prob_mean = mean(temp);
        prob_std = std(temp);
        prop_all(ll,ii) = prob_mean;
        disp(['--> Group ',num2str(ll),' : ',num2str(prob_mean),' with std ',num2str(prob_std)])
        h=histogram(temp,'Normalization','probability','FaceAlpha',FaceAlpha,'FaceColor',color_order(ll,:));
        h.NumBins = BinsN;
        ylim([0 ymax])
        xlim([0 1])
        set(gca,'yticklabel',[],'ytick',[],'FontSize',16,'fontname','times')
        name=['State ',num2str(ii)];
        title(name,'FontSize',20,'fontname','times','FontWeight','normal')
        hold on;
        Ylim=ylim;
        line([mean(temp) mean(temp)],[Ylim(1) Ylim(2)],...
            'linestyle','-.', 'Color',color_order(ll,:), 'LineWidth', 1.5);
    end
    if ii==1
        ylabel('$\lambda^0_x$','Interpreter','latex','FontSize',20)
    end
    legend(legend_name,'FontSize',14,'fontname','times') % ,'Location','northwest'
    legend('boxoff')
end
disp('============> Remaining probability ')
for ll=1:4 % level
    disp(['--> Group ',num2str(ll),' : ',num2str(1-sum(prop_all(ll,:)))])
end

% density functions %
xmin=-4;
xmax=6;
ymax=1.2;
yy = linspace(xmin,xmax,100); 
legend_name={'YM','YF','EM','EF'};

for jj=1:length(NA_state)
    nexttile(ii+jj)
    ss = NA_state(jj);
    for ll=1:4
        temp_mu=mu(state_order(ss),:);
        temp_sigma=sqrt(sigma(state_order(ss),:));
        temp_mu=temp_mu+ll*beta;

        pp = normpdf(yy,temp_mu',temp_sigma');
        p = mean(pp,1);
        %%% CI %%%
        if CI_method==1 % method 1: t-distribution
            p_std = std(pp,1);
            ssz=length(temp_mu);
            p_low = p - tinv(0.975,ssz-1)*(p_std/sqrt(ssz));
            p_up = p + tinv(0.975,ssz-1)*(p_std/sqrt(ssz));
        end
        if CI_method==2 % method 2: percentile
            p_low=prctile(pp,2.5,1);
            p_up=prctile(pp,97.5,1);
        end 
        plot(yy,p,'LineWidth', 2,'Color',color_order(ll,:)); 
        xlim([xmin xmax])
        ylim([0 ymax])
        set(gca,'yticklabel',[],'ytick',[],'FontSize',16,'fontname','times')
        hold on
        if CI==1
            h_temp=fill([yy,fliplr(yy)],[p_low,fliplr(p_up)],color_order(ll,:));
            set(h_temp,'edgealpha',0,'facealpha',0.4)
            legend_name={'YM','95% CI','YF','95% CI',...
                'EM','95% CI','EF','95% CI'};
        end
    end
    if jj==1
        ylabel('Density','FontSize',20)
    end
    legend(legend_name,'FontSize',14,'fontname','times') % ,'Location','northwest'
    legend('boxoff')
end
disp('===================================================================')
