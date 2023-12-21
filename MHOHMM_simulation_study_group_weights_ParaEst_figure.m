

clear;clc;close all;

%% Load posterior samples
% seed 1: model 1
% seed 5: model 2
% seed 6: model 2
% seed 18: model 1 & 2 (Y)
% seed 21: model 1 & 2
% seed 23: model 1 & 2 (Y)
% seed 24: model 1


seed=23;
mm=2; % model selection

filename=['Result_AOAS/Parameter_estimation/selected/ParaEst_ForFigure_Seed_',num2str(seed),'.mat'];
% filename=['Result_AOAS/history/ParaEst_ForFigure_Seed_',num2str(seed),'.mat'];
load(filename) 
%'lambda_x0_figure','mu_figure','sigma_figure','beta_figure','Sigma_r_figure')

%% Parameters

%%% True Parameters %%%
if seed==18
% seed 18 model 1
lambda0_N_A=[0.688386286180819;0.133183797512717;0.178429916306464];
lambda0_N_B=[0.366007598550494;0.24702190171412;0.386970499735386];
elseif seed==23
% seed 23 model 1
lambda0_N_A=[0.141285763007481;0.334343961417762;0.524370275574757];
lambda0_N_B=[0.391710786532751;0.192461015628071;0.415828197839178];
elseif seed==1
% seed 1 model 1
lambda0_N_A=[0.319881973230968;0.215419938802168;0.464698087966864];
lambda0_N_B=[0.759906669938898;0.100203179746648;0.139890150314453];
elseif seed==21
% seed 21 model 1
lambda0_N_A=[0.469380912439095;0.188676638917642;0.341942448643263];
lambda0_N_B=[0.257084746785157;0.543972744534209;0.198942508680634];
elseif seed==24
% seed 24 model 1
lambda0_N_A=[0.182510571349534;0.625046666843238;0.192442761807227];
lambda0_N_B=[0.177426143047816;0.544101183258607;0.278472673693578];

elseif seed==2
    if mm==1
       lambda0_N_A=[0.639251384326772;0.0970855216672375;0.26366309400599];  
       lambda0_N_B=[0.656601361952897;0.115877063099499;0.227521574947605];
    elseif mm==2
       lambda0_NA_A=[0.624624474454629;0.275425622339441;0.0999499032059304];  
       lambda0_NA_B=[0.789211029185831;0.118787436714139;0.0920015341000299];
    end
    mu0_N=[0,2,4]; % model 1: 'Normal'
    mu0_NA=[0,2,4]; % model 2: 'Abnormal'
    beta_N0=1;
    beta_NA0=1;
    sigma0_R=0.5;
    sigma0_N=0.5;
    sigma0_NA=0.5;
end

% round 
if mm==1
lambda0_N_A=round(lambda0_N_A,2);
lambda0_N_B=round(lambda0_N_B,2);    
elseif mm==2
lambda0_NA_A=round(lambda0_NA_A,2);
lambda0_NA_B=round(lambda0_NA_B,2);
end

%%% state order %%%
% level 1
temp_mu=mu_figure{1,8}; % state X samples
mu_mean=mean(temp_mu,2);
mu_mean_order=sort(mu_mean);
state_order1=1:3;
for ii=1:3
    state_order1(ii)=find(mu_mean==mu_mean_order(ii));
end

% level 2
temp_mu=mu_figure{2,8}; % state X samples
mu_mean=mean(temp_mu,2);
mu_mean_order=sort(mu_mean);
state_order2=1:3;
for ii=1:3
    state_order2(ii)=find(mu_mean==mu_mean_order(ii));
end
state_order=[state_order1;state_order2];

Model_Name={'Hf','Hfr','Hf.Of','Hfr.Of','Hf.Or','Hfr.Or','Hf.Ofr','Hfr.Ofr'};

%% Figure 

order=state_order(mm,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% State prevalence probabilities %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% str=3:8; % model structure selection
% str=[3,4,7,8];
str=7:8;
str_Num=length(str);

legend_name=cell(1,str_Num+1);
legend_name(1:str_Num)=Model_Name(str);
legend_name{end}='True';

color_order = get(gca,'colororder');


%%% Level 1 %%%
ymax=0.15;
BinsN=36;

tiledlayout(2,3,'TileSpacing','tight'); % loose; compact; tight; none

for ii=1:3 % 3 states
    nexttile(ii)
    for ss=1:str_Num
        lambda_x0=lambda_x0_figure{mm,str(ss)};
        temp1=lambda_x0(order(ii),1,:);
        h=histogram(temp1,'Normalization','probability','FaceAlpha',0.4,'FaceColor',color_order(ss,:)); %
        % h.BinWidth = 0.05;
        h.NumBins = BinsN;
        ylim([0 ymax])
%         xlim([0 1])
        set(gca,'yticklabel',[],'ytick',[],'FontSize',16,'fontname','times')
        if mm==1
            name=['$\lambda_1^0$(',num2str(ii),')=',num2str(lambda0_N_A(ii))];
        elseif mm==2
            name=['$\lambda_1^0$(',num2str(ii),')=',num2str(lambda0_NA_A(ii))];
        end
        if ii==1
            ylabel('$x=1$','Interpreter','latex','FontSize',24)
        end
        title(name,'Interpreter','latex','FontSize',20)
        hold on;
    end
    
    Ylim=ylim;
    if mm==1
        line([lambda0_N_A(ii) lambda0_N_A(ii)],[Ylim(1) Ylim(2)],...
            'linestyle','-.', 'Color','r', 'LineWidth', 3);
    elseif mm==2
        line([lambda0_NA_A(ii) lambda0_NA_A(ii)],[Ylim(1) Ylim(2)],...
            'linestyle','-.', 'Color','r', 'LineWidth', 3);
    end
    
    legend(legend_name,'FontSize',14,'fontname','times') % ,'Location','northwest'
    legend('boxoff')
end
    
%%% Level 2 %%%

for ii=1:3 % 3 states
    nexttile(ii+3)
    for ss=1:str_Num
        lambda_x0=lambda_x0_figure{mm,str(ss)};
        temp1=lambda_x0(order(ii),2,:);
        h=histogram(temp1,'Normalization','probability','FaceAlpha',0.4,'FaceColor',color_order(ss,:)); % 
        % h.BinWidth = 0.05;
        h.NumBins = BinsN;
        ylim([0 ymax])
%         xlim([0 1])
        set(gca,'yticklabel',[],'ytick',[],'FontSize',16,'fontname','times')
        if mm==1
            name=['$\lambda_2^0$(',num2str(ii),')=',num2str(lambda0_N_B(ii))];
        elseif mm==2
            name=['$\lambda_2^0$(',num2str(ii),')=',num2str(lambda0_NA_B(ii))];
        end
        if ii==1
            ylabel('$x=2$','Interpreter','latex','FontSize',24)
        end
        title(name,'Interpreter','latex','FontSize',20)
        hold on;
    end
    
    Ylim=ylim;
    if mm==1
        line([lambda0_N_B(ii) lambda0_N_B(ii)],[Ylim(1) Ylim(2)],...
            'linestyle','-.', 'Color','r', 'LineWidth', 3);
    elseif mm==2
        line([lambda0_NA_B(ii) lambda0_NA_B(ii)],[Ylim(1) Ylim(2)],...
            'linestyle','-.', 'Color','r', 'LineWidth', 3);
    end
    legend(legend_name,'FontSize',14,'fontname','times') % ,'Location','northwest'
    legend('boxoff')
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Emission distribution density %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% str=[3,4,7,8]; % model structure selection
str=8;
str_Num=length(str);

legend_name=cell(1,2*str_Num+1);
for t=1:str_Num
    legend_name((t-1)*2+1)=Model_Name(str(t));
    if 1==0
        temp=Model_Name(str(t));
        temp2=[temp{1},' (95% CI)'];
    end
    if 1==1
        temp2='95% CI';
    end    
    legend_name{(t-1)*2+2}=temp2;
end
legend_name{end}='True';

marker={'o','*','s','d','^','p'};
maker_indice=5;

%%%%%%%%%% figures %%%%%%%%%%
yy1 = linspace(-4,4,100); % state 1 (mu = 0)
yy=cell(3,2); % 3 states X 2 levels
yy{1,1}=yy1+1;
yy{2,1}=yy1+1+2;
yy{3,1}=yy1+1+4;
yy{1,2}=yy1+2;
yy{2,2}=yy1+2+2;
yy{3,2}=yy1+4+2;

figure;
tiledlayout(2,3,'TileSpacing','compact'); % loose; compact; tight; none

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Marginalize out random effects %%%
%%%        Using all samples       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if 1==1
%%% Level 1 %%%
for ii=1:3 % 3 states
    nexttile(ii)
    for ss=1:str_Num
        temp_mu=mu_figure{mm,str(ss)}; 
        temp_sigma=sigma_figure{mm,str(ss)}; 
        temp_sigmaR=Sigma_r_figure{mm,str(ss)}; 
        temp_beta=beta_figure{mm,str(ss)}; 
        mu=temp_mu(order(ii),:)+temp_beta;
        sigma=sqrt(temp_sigma(order(ii),:)+temp_sigmaR);
        pp = normpdf(yy{ii,1},mu',sigma'); % samples X n
        p = mean(pp,1);
        
        %%% CI %%%
        if 1==0 % method 1: t-distribution
            p_std = std(pp,1);
            ssz=length(mu);
            p_low = p - tinv(0.975,ssz-1)*(p_std/sqrt(ssz));
            p_up = p + tinv(0.975,ssz-1)*(p_std/sqrt(ssz));
        end
        if 1==1 % method 2: percentile
            p_low=prctile(pp,2.5,1);
            p_up=prctile(pp,97.5,1);
        end
        
        x=yy{ii,1};
        if mm==1
            p_true = normpdf(x,mu0_N(ii)+beta_N0,sqrt(sigma0_N^2+sigma0_R^2));
        elseif mm==2
            p_true = normpdf(x,mu0_NA(ii)+beta_NA0,sqrt(sigma0_NA^2+sigma0_R^2));
        end
        plot(x,p,'LineWidth', 2,'Color',color_order(ss,:)); % ,'Marker',marker{ss},'MarkerIndices',1:maker_indice:length(p)
        set(gca,'yticklabel',[],'ytick',[],'FontSize',16,'fontname','times')
        if ii==1
            ylabel('$x=1$','Interpreter','latex','FontSize',24)
        end
        name=['State ',num2str(ii)];
        title(name,'FontSize',16,'FontWeight','Normal')
        hold on
        h_temp=fill([x,fliplr(x)],[p_low,fliplr(p_up)],color_order(ss,:));
        set(h_temp,'edgealpha',0,'facealpha',0.4)
    end
    plot(x,p_true,'r-.', 'LineWidth', 3);
    legend(legend_name,'FontSize',14,'fontname','times') % ,'Location','northwest'
    legend('boxoff')
end

%%% Level 2 %%%
for ii=1:3 % 3 states
    nexttile(ii+3)
    for ss=1:str_Num
        temp_mu=mu_figure{mm,str(ss)}; 
        temp_sigma=sigma_figure{mm,str(ss)}; 
        temp_sigmaR=Sigma_r_figure{mm,str(ss)}; 
        temp_beta=beta_figure{mm,str(ss)}; 
        mu=temp_mu(order(ii),:)+2*temp_beta;
        sigma=sqrt(temp_sigma(order(ii),:)+temp_sigmaR);
        pp = normpdf(yy{ii,2},mu',sigma'); % samples X n
        p = mean(pp,1);
        %%% CI %%%
        if 1==0 % method 1: t-distribution
            p_std = std(pp,1);
            ssz=length(mu);
            p_low = p - tinv(0.975,ssz-1)*(p_std/sqrt(ssz));
            p_up = p + tinv(0.975,ssz-1)*(p_std/sqrt(ssz));
        end
        if 1==1 % method 2: percentile
            p_low=prctile(pp,2.5,1);
            p_up=prctile(pp,97.5,1);
        end 
        x=yy{ii,2};
        if mm==1
            p_true = normpdf(x,mu0_N(ii)+2*beta_N0,sqrt(sigma0_N^2+sigma0_R^2));
        elseif mm==2
            p_true = normpdf(x,mu0_NA(ii)+2*beta_NA0,sqrt(sigma0_NA^2+sigma0_R^2));
        end
        plot(x,p,'LineWidth', 2,'Color',color_order(ss,:)); % ,'Marker',marker{ss},'MarkerIndices',1:maker_indice:length(p)
        set(gca,'yticklabel',[],'ytick',[],'FontSize',16,'fontname','times')
        if ii==1
            ylabel('$x=2$','Interpreter','latex','FontSize',24)
        end
        name=['State ',num2str(ii)];
        title(name,'FontSize',16,'FontWeight','Normal')
        hold on
        h_temp=fill([x,fliplr(x)],[p_low,fliplr(p_up)],color_order(ss,:));
        set(h_temp,'edgealpha',0,'facealpha',0.4)
    end
    plot(x,p_true,'r-.', 'LineWidth', 3);
    legend(legend_name,'FontSize',14,'fontname','times') % ,'Location','northwest'
    legend('boxoff')
end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Marginalize out random effects %%%
%%%      Using posterior mean      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if 1==0
%%% Level 1 %%%
for ii=1:3 % 3 states
    nexttile(ii)
    for ss=1:str_Num
        temp_mu=mu_figure{mm,str(ss)}; 
        temp_mu=mean(temp_mu);
        temp_sigma=sigma_figure{mm,str(ss)}; 
        temp_sigma=mean(temp_sigma);
        temp_sigmaR=Sigma_r_figure{mm,str(ss)}; 
        temp_sigmaR=mean(temp_sigmaR);
        p = normpdf(yy{ii,1},temp_mu(order(ii)),sqrt(temp_sigma(order(ii))+temp_sigmaR));
        if mm==1
            p_true = normpdf(yy{ii,1},mu0_N(ii),sqrt(sigma0_N^2+sigma0_R^2));
        elseif mm==2
            p_true = normpdf(yy{ii,1},mu0_NA(ii),sqrt(sigma0_NA^2+sigma0_R^2));
        end
        plot(yy{ii,1},p,'LineWidth', 2); % ,'Color',color_order(ss,:),'Marker',marker{ss},'MarkerIndices',1:maker_indice:length(p)
        set(gca,'yticklabel',[],'ytick',[])
        hold on
    end
    plot(yy{ii,1},p_true,'r-.', 'LineWidth', 3);
    legend(legend_name,'FontSize',12) % ,'Location','northwest'
    legend('boxoff')
end

%%% Level 2 %%%
for ii=1:3 % 3 states
    nexttile(ii+3)
    for ss=1:str_Num
        temp_mu=mu_figure{mm,str(ss)}; 
        temp_mu=mean(temp_mu);
        temp_sigma=sigma_figure{mm,str(ss)}; 
        temp_sigma=mean(temp_sigma);
        temp_sigmaR=Sigma_r_figure{mm,str(ss)}; 
        temp_sigmaR=mean(temp_sigmaR);
        temp_beta=beta_figure{mm,str(ss)}; 
        temp_beta=mean(temp_beta);
        p = normpdf(yy{ii,1},temp_mu(order(ii))+temp_beta,sqrt(temp_sigma(order(ii))+temp_sigmaR));
        if mm==1
            p_true = normpdf(yy{ii,2},mu0_N(ii)+beta_N0,sqrt(sigma0_N^2+sigma0_R^2));
        elseif mm==2
            p_true = normpdf(yy{ii,2},mu0_NA(ii)+beta_NA0,sqrt(sigma0_NA^2+sigma0_R^2));
        end
        plot(yy{ii,2},p,'LineWidth', 2); % ,'Color',color_order(ss,:),'Marker',marker{ss},'MarkerIndices',1:maker_indice:length(p)
        set(gca,'yticklabel',[],'ytick',[])
        hold on
    end
    plot(yy{ii,2},p_true,'r-.', 'LineWidth', 3);
    legend(legend_name,'FontSize',12) % ,'Location','northwest'
    legend('boxoff')
end
end








