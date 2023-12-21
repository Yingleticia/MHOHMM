clear;clc;close all;
%%%%%% Hard threshold definition of AHE
%%%%%% within 30 min, 90% measurements of MAP<60 mmHg
%%%%%% no more than 10% missing data in the prediction window
%%%%%% at least 5-hour recording (360-min)

partNum=2;

N_ahead=275; % 4-hour ahead 

% load record info
excelName=['AHE_P01_info_part',num2str(partNum),'.xlsx'];
temp = readtable(excelName);
ID_all=temp{:,1}; % ID
record_all=temp{:,2}; % Record name
ICU=temp{:,3}; % ICU stay
Gender=temp{:,4}; % Gender
Age=temp{:,5}; % Age
SignalNum=temp{:,8}; % length of the signal
ff=temp{:,7}; % frequency
idx_all=temp{:,10:11}; % 1: ASBP 2: ADBP 


% Output 
UseLabel=zeros(size(idx_all,1),1); % 1: AHE -1: cannot be used 0: Non-AHE 
Dat_Select=cell(1,size(idx_all,1)); % collect data for case study

% denoise parameter setting
R=2; % median+/-R*IQR
b_cut=20; % median+/-b_cut (MAP)
win_size=5;
Max_length=10; % consecutive missing values

kk=5; % 34: denoise (median-pass); 
figure_type=2; % 1: denoise (median-pass); 2: AHE definition

    
if SignalNum(kk)<300 % at least 5-hour measurements
    UseLabel(kk)=-1;
else
    %%%%%%% load raw data %%%%%%% 
    record_temp=record_all(kk);
    record=record_temp{1}; % record name
    record_name=['mimic3wdb/matched/',record];
    idx=idx_all(kk,:); % index for the vital signs 1: ASBP 2: ADBP 
    [rawDat,~,~]=rdsamp(record_name);
    rawDat=rawDat(:,idx);

    %%%%%%% Remove erroneous values %%%%%%%
    % 1: ASBP
    rawDat(rawDat(:,1)<40,1)=nan;
    rawDat(rawDat(:,1)>180,1)=nan;
    % 2: ADBP
    rawDat(rawDat(:,2)<30,2)=nan;
    rawDat(rawDat(:,2)>160,2)=nan;

    %%%%%%% Frequency change: per min %%%%%%
    if ff(kk)~=1 % data are collected in a second
        sizeMin=ceil(size(rawDat,1)/60); % sequence length (in minute)
        Dat_temp=zeros(sizeMin,length(idx)); % data with measurements in minutes
        for tt=1:sizeMin
            if tt==sizeMin
                DataSegment=rawDat(((tt-1)*60+1):end,:);
            else
                DataSegment=rawDat(((tt-1)*60+1):(tt*60),:);
            end

            for jj=1:length(idx)
                vital_jj=DataSegment(:,jj);
                Dat_temp(tt,jj)=mean(vital_jj(~isnan(vital_jj)));
            end
        end
    else
        Dat_temp=rawDat;
    end

    %%%%%%% Compute: MAP %%%%%%
    MAP_all=(2/3)*Dat_temp(:,2)+(1/3)*Dat_temp(:,1); % MAP=2/3 DBP+1/3 SBP 
    MAP_all(((MAP_all<40)+(MAP_all>160))>0)=nan; % data before denoise

    %%%%%%% Remove: consecutive missing values over long period %%%%%%
    NaN_idx=isnan(MAP_all); % idx for nan
    idx_start=find(NaN_idx==0,1,'first'); % starting point for non-nan value
    idx_end=find(NaN_idx==0,1,'last'); % starting point for non-nan value
    MAP_temp=MAP_all(idx_start:idx_end);
    NaN_idx=isnan(MAP_temp); % idx for nan
    
    if 1==0
        figure(1);
        plot(MAP_temp); % raw signal
    end

    %%%%%%% Remove: Outliers (Median-pass) %%%%%%
    idxMiss=find(isnan(MAP_temp));
    signal=MAP_temp; % missing data imputation with previous valid value
    if ~isempty(idxMiss)
        for tt=1:length(idxMiss)
            signalAhead=signal(1:(idxMiss(tt)-1));
            signal(idxMiss(tt))=signalAhead(end);
        end
    end
    % (Median-pass)
    for ii=win_size:length(signal)
        dat_win=signal((ii-win_size+1):ii);
        med=median(dat_win);
        IQR=iqr(dat_win);
        temp1=abs(signal(ii)-med)>IQR*R;
        temp2=IQR>1e-6;
        temp3=abs(signal(ii)-med)>b_cut;
        NoMissIdx=~NaN_idx((ii-win_size+1):ii);
%         temp4=sum(NoMissIdx)>0;
        temp4=true;
        % constraint: no so many missing data within the window
        if ((temp1 && temp2) || temp3) && temp4
            NoMissDat=dat_win(~NaN_idx((ii-win_size+1):ii));
            signal(ii)=NoMissDat(end);
            NaN_idx(ii)=true; % identified as outlier
        end
    end  
    MAP_denoise=signal; % MAP after denoise
    
    if 1==0
        figure(2);
        plot(MAP_denoise); % raw signal
    end

    %%%%%%% Event detection %%%%%%
    if sum(~NaN_idx)<300
         UseLabel(kk)=-1;
    else
       %%% Label the MAP %%% 
       label=MAP_denoise<60; % label for low BP (Hypotension) 

       %%% events detection %%%
       sz=size(label,1); % # of observations
       BP_low=[]; % detect AHE
       NormalUse=[]; % detect Non-AHE
       i=1;
       while isempty(BP_low) && i<=(sz-N_ahead-30+1) % keep the first detected AHE
           nan_temp=NaN_idx(i:(i+N_ahead+30-1));
           % detect consecutive missing values 
           mm=find(nan_temp==1,1,'first');
           start_new=i+1; % next idx for i
           MissConstraint1=true; 
           while ~isempty(mm)
               nan_temp=nan_temp(mm:end);
               nn=find(nan_temp==0,1,'first');
               if ~isempty(nn)
                   length_temp=nn-1;
                   if length_temp>=Max_length
                       start_new=i+(N_ahead+30-length(nan_temp(nn:end)));
                       MissConstraint1=false;
                       mm=[];
                   else
                       nan_temp=nan_temp(nn:end);
                       mm=find(nan_temp==1,1,'first');
                   end
               else
                   length_temp=length(nan_temp);
                   if length_temp>=Max_length
                       start_new=i+(N_ahead+30);
                       MissConstraint1=false;
                       mm=[];
                   else
                       mm=[];
                   end
               end
           end

           if MissConstraint1 % consecutive missing values less than 10 min
               MissConstraint2=sum(NaN_idx(i:(i+N_ahead+30-1)))<=(0.1*(N_ahead+30)); 
               dat_temp=MAP_denoise(i:(i+N_ahead+30-1));
               std_temp=std(dat_temp);
               varCondition=std_temp>1; % remove consecutive equal values
               if MissConstraint2 && varCondition % missing data less than 10% && remove equal values for long period
                   if sum(label((i+N_ahead):(i+N_ahead+29)))>=0.9*30 % AHE
                       BP_low=i;
                       UseLabel(kk)=1;
                   elseif sum(label(i:(i+N_ahead+29)))<=(0.1*(N_ahead+30)) && isempty(NormalUse)
                       NormalUse=i;
                   end
               end
               i=start_new; 
           else
               i=start_new; 
           end
       end

    end

end

%%%%%%% Figures %%%%%%
if figure_type==1
    if UseLabel(kk)==1 % save observations ahead if AHE
        Dat_Select{kk}=MAP_denoise(BP_low:(BP_low+N_ahead+29+60)); % signal length: N_ahead+30=270

        Dat_kk=Dat_Select{kk};
        Dat_kk_final=Dat_kk((win_size+1):end);
        R=3; % median+/-R*IQR
        b_cut=40; % median+/-b_cut (MAP)

        figure(3);
        subplot(2,1,1);
        plot(Dat_kk_final); hold on;
        xmin=0;
        xmax=length(Dat_kk_final);
        xlim([xmin xmax]);
        ylim([40 100]);
    %         line([0 xmax],[60 60]);

        for ii=1:(N_ahead-win_size+30)
            dat_temp=Dat_kk(ii:(ii+win_size));
            med=median(dat_temp);
            IQR=iqr(dat_temp);
            temp1=abs(Dat_kk_final(ii)-med)>IQR*R;
            temp2=IQR>1e-6;
            temp3=abs(Dat_kk_final(ii)-med)>b_cut;
            if (temp1 && temp2) || temp3
                Dat_kk_final(ii)=Dat_kk(ii+win_size-1);
                Dat_kk(ii+win_size)=Dat_kk(ii+win_size-1);
            end
        end

        figure(3);
        subplot(2,1,2);
        plot(Dat_kk_final); hold on;
        xlim([xmin xmax]);
        ylim([40 100]);
    %         line([0 xmax],[60 60]);
    %         line([271 271],[40 100]);

        figure(4);
        plot(Dat_kk_final); hold on;
        xlim([xmin xmax]);
        ylim([40 100]);
        line([0 xmax],[60 60]);

        line([271 271],[40 100]);
        line([240 240],[40 100]);
        line([120 120],[40 100]);

    end
elseif figure_type==2
    if UseLabel(kk)==1 % save observations ahead if AHE
        Dat_Select{kk}=MAP_denoise(BP_low:(BP_low+N_ahead+29+150)); 

        Dat_kk=Dat_Select{kk};
        Dat_kk_final=Dat_kk((win_size+1):end);
        R=3; % median+/-R*IQR
        b_cut=40; % median+/-b_cut (MAP)
        xmin=0;
        xmax=length(Dat_kk_final);

        for ii=1:(N_ahead-win_size+30)
            dat_temp=Dat_kk(ii:(ii+win_size));
            med=median(dat_temp);
            IQR=iqr(dat_temp);
            temp1=abs(Dat_kk_final(ii)-med)>IQR*R;
            temp2=IQR>1e-6;
            temp3=abs(Dat_kk_final(ii)-med)>b_cut;
            if (temp1 && temp2) || temp3
                Dat_kk_final(ii)=Dat_kk(ii+win_size-1);
                Dat_kk(ii+win_size)=Dat_kk(ii+win_size-1);
            end
        end

        figure(4);
        plot(Dat_kk_final); hold on;
        xlim([xmin xmax]);
        ylim([40 100]);
        line([0 xmax],[60 60]);
        
        line([270 270],[40 100]);
        line([240 240],[40 100]);
        line([120 120],[40 100]);
        
    end
end
fprintf('Patient %i has uselabel %i\n', kk,UseLabel(kk));



