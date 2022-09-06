% setups
clc; close all; clear all;
addpath 'D:\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG\eeglab2021.0';
%addpath 'D:\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG\Data';
%addpath (genpath ('D:\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG\Data'));
%addpath (genpath ('D:\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG\CSDtoolbox'));
ch_labels = {'Fp1';'Fz';'F3';'F7';'FT9';'FC5';'FC1';'C3';'T7';'TP9';'CP5';'CP1';'Pz';'P3';'P7';'O1';'Oz';'O2';'P4';'P8';'TP10';'CP6';'CP2';'Cz';'C4';'T8';'FT10';'FC6';'FC2';'F4';'F8';'Fp2'};
Corrected_ch_labels = {'AFF3h';'AFF5h';'F7';'AFF1';'FFC5h';'FFC3h';'FCC5h';'FCC3h';'CCP5h';'CCP3h';'P3';'P5';'PO7';'P7';'PO3';'FT7';'AFF4h';'AFF6H';'F8';'AFF2';'FFC6h';'FFC4h';'FCC6h';'FCC4h';'CCP6h';'CCP4h';'P4';'P6';'PO8';'P8';'PO4';'FT8'};

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;        % initialize eeglab
%% preprocessing pipline
nos.Exp = 15;  % 15 -> FLS suturing, first sub has some problem
nos.Nov = 15;  % 15 -> FLS suturing
tic;
for m = 1:1  % 1 for Exp and 2 for Nov
    if m == 1
        N = nos.Exp;
    else
        N = nos.Nov;
    end
    c =  13;
    for s = 15:N %N  % N   no of Subjects (N)
        sub = [];
        EEG = [];
        ALLEEG = [];
        CURRENTSET = [];
        if m == 1
            if s == 1
                c = 7;
            end
            sub = sprintf('C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG\\Data_EEG_fNIRS_Suturing\\EEG_raw - Copy (2)\\E0%d\\',c);
            sub_data = sprintf('E0%dproc.set',c);
            save_filename = sprintf('E0%dproc.set',c);
            sub_data_snirf = sprintf('E0%d.snirf',c);
            sub_stimuli = sprintf('E0%d',c);
            sub_trial_name = sprintf('Exp_%d_Exp',c);
            hdrfile = sprintf('E0%d.vhdr',c);
        else
            sub = sprintf('C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG\\Data_EEG_fNIRS_Suturing\\EEG_raw - Copy (2)\\N0%d\\',c);
            sub_data = sprintf('N0%dproc.set',c);
            save_filename = sprintf('N0%dproc.set',c);
            sub_data_snirf = sprintf('N0%d.snirf',c);
            sub_stimuli = sprintf('N0%d',c);
            sub_trial_name = sprintf('Nov_%d_Nov',c);
            hdrfile = sprintf('N0%d.vhdr',c);
        end
        % preprocessing steps
        fprintf('\n ####################################### Analyzing %s ###########################################\n',hdrfile);
        %[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
        if m == 2 && s == 2 
            EEG = pop_loadset('filename','N02.set','filepath',sub);
            data1 = EEG.data(1:32,:);
            EEG.data = [];
            EEG.data = data1;
            data1 = [];
            [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
            EEG = eeg_checkset( EEG );
%             EEG = pop_rmbase( EEG, [],[]);  % remove epoch baseline;
%             removing baseine for one epoch(considered here to ICA
%             decomposition) ca have dramatic effect on ICA; EEGlab 
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'overwrite','on','gui','off'); 
            EEG = eeg_checkset( EEG );
            EEG = pop_reref( EEG, []);      % reference
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'overwrite','on','gui','off'); 
            EEG = pop_eegfiltnew(EEG, 'locutoff',1,'minphase',1,'plotfreqz',0);             % high pass filter at 1Hz (lower end)
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','','overwrite','on','gui','off');
            EEG = pop_cleanline(EEG, 'bandwidth',2,'chanlist',[1:32] ,'computepower',1,'linefreqs',60,'newversion',0,'normSpectrum',0,'p',0.01,'pad',2,'plotfigures',0,'scanforlines',0,'sigtype','Channels','taperbandwidth',2,'tau',100,'verb',1,'winsize',4,'winstep',1);
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'overwrite','on','gui','off'); 
            EEG = eeg_checkset( EEG );
            EEG = pop_reref( EEG, []);      % reference again
            %Current source density (CSD) algorithm for spatial filtering
%             M = ExtractMontage('10-5-System_Mastoids_EGI129.csd',Corrected_ch_labels);
%             [G,H] = GetGH(M);
% %             Data_Channel = CSD(EEG.data, G, H, 0.00001,8.5);
%             EEG.data = Data_Channel;
        else 
            EEG = pop_loadbv(sub, hdrfile,[], [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]);
            %EEG = pop_loadbv('D:\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG\Data\EEG_raw - Copy (2)\N02\','N02.vhdr',[], [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]);
            % 'D:\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG\Data\EEG_raw - Copy (2)\E02\'
            [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
            EEG = eeg_checkset( EEG );
%             EEG = pop_rmbase( EEG, [],[]);
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'overwrite','on','gui','off');
            EEG = eeg_checkset( EEG );
            EEG = pop_reref( EEG, []);      % reference
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'overwrite','on','gui','off');
            EEG = pop_eegfiltnew(EEG, 'locutoff',1,'minphase',1,'plotfreqz',0);             % high pass filter at 1Hz (lower end)
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','','overwrite','on','gui','off');
            EEG = pop_cleanline(EEG, 'bandwidth',2,'chanlist',[1:32] ,'computepower',1,'linefreqs',60,'newversion',0,'normSpectrum',0,'p',0.01,'pad',2,'plotfigures',0,'scanforlines',0,'sigtype','Channels','taperbandwidth',2,'tau',100,'verb',1,'winsize',4,'winstep',1);
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'overwrite','on','gui','off'); 
            EEG = eeg_checkset( EEG );
            EEG = pop_reref( EEG, []);      % reference again
            %Current source density (CSD) algorithm for spatial filtering
%             M = ExtractMontage('10-5-System_Mastoids_EGI129.csd',Corrected_ch_labels);
%             [G,H] = GetGH(M);
%             Data_Channel = CSD(EEG.data, G, H, 0.00001,8.5);
%             EEG.data = Data_Channel;
        end
        EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','off');
        [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
        EEG = eeg_checkset( EEG );
        %EEG = pop_saveset( EEG, 'filename',save_filename,'filepath','C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG\\Data_EEG_fNIRS_Suturing\\EEG_preprocessed_sept27\\');
                                                                    
        [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

        %[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'savenew',save_filename,'overwrite','on','gui','off'); 
%         %preprocessing steps
%         tmpdata = EEG.data;
%         tmprank = rank(tmpdata);    % matlab func for rank
%         covarianceMatrix = cov(tmpdata', 1);
%         [E, D] = eig (covarianceMatrix);
%         rankTolerance = 1e-7;
%         tmprank2=sum (diag (D) > rankTolerance);
%         Correlation = corrcoef(tmpdata');
%         % correlation
%         h = heatmap(Correlation,'Colormap', jet,'FontSize',10);
%         h.XDisplayLabels = ch_labels;
%         h.YDisplayLabels = ch_labels;
%         h.Title = 'Correlation of E03';
%         xlabel('Channel Name')
%         ylabel('Channel Name')
%         set(struct(h).NodeChildren(3), 'XTickLabelRotation', 45);
%         if tmprank ~= tmprank2
%             fprintf('Warning: fixing rank computation inconsistency (%d vs %d) most likely because running under Linux 64-bit Matlab\n', tmprank, tmprank2);
%             %tmprank2 = max(tmprank, tmprank2);
%             tmprank2 = min(tmprank, tmprank2);
%         end
%         fprintf('Analyzing %s \n',sub_trial_name);
        c = c+1;
    end
end
toc;