clear
close all


%%


path_clean = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/clean_trainset_wav_16k/';
path_noisy = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/noisy_trainset_wav_16k/';

dest_clean = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/mat_clean_16k/';
dest_noisy = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/mat_noisy_16k/';

%% Main parameters

% fs_new=16000;
chunk_size = 1000; % Each chunk_size wav files would be a single mat file 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CLEAN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Folder and File names

% wav_path=path_clean;  
% wav_names=dir(wav_path);
% wav_names=extractfield(wav_names,'name');
% wav_names=wav_names(3:end);
% num_wav_files=length(wav_names);
% 
% %% Body
% 
% for i=0 : floor(num_wav_files/chunk_size)-1
%     
%     display(i, 'mat file number')
%     
%     concat_wav=[];
%     
%     for j=1:chunk_size
%         
%         display(j, 'wav file number')
% 
%         [temp_wav, fs]=audioread([wav_path, wav_names{ i*chunk_size + j }]);
% 
%     %     temp_wav=resample(temp_wav,fs_new, fs);
% 
%         concat_wav=[concat_wav, temp_wav'];
% 
%     end
% 
%     %% removing noise
%         
% %     concat_wav_raw=concat_wav;
%     %     concat_wav=concat_wav_raw;
%     %     concat_wav( abs(concat_wav) > 50*median(abs(concat_wav)))=0;
% 
%     %% Scaling signal
%     maxi=max(concat_wav);
%     mini=min(concat_wav);
%     % concat_wav=concat_wav+abs(min(concat_wav));
%     concat_wav=concat_wav - mean(concat_wav);
%     if maxi>abs(mini)
%         up_lim=maxi;
%     else
%         up_lim=abs(mini);
%     end
%     concat_wav=concat_wav/up_lim*0.9;
%     
%     %% Saving
%     save_path = [dest_clean, 'clean_', num2str(i), '.mat'];
%     save(save_path,'concat_wav') %-v7.3 
% 
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NOISY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Folder and File names

wav_path=path_noisy;  
wav_names=dir(wav_path);
wav_names=extractfield(wav_names,'name');
wav_names=wav_names(3:end);
num_wav_files=length(wav_names);

%% Body

for i=0 : floor(num_wav_files/chunk_size)-1
    
    display(i, 'mat file number')
    
    concat_wav=[];
    
    for j=1:chunk_size
        
        display(j, 'wav file number')

        [temp_wav, fs]=audioread([wav_path, wav_names{ i*chunk_size + j }]);

    %     temp_wav=resample(temp_wav,fs_new, fs);

        concat_wav=[concat_wav, temp_wav'];

    end

    %% removing noise
        
%     concat_wav_raw=concat_wav;
    %     concat_wav=concat_wav_raw;
    %     concat_wav( abs(concat_wav) > 50*median(abs(concat_wav)))=0;

    %% Scaling signal
    maxi=max(concat_wav);
    mini=min(concat_wav);
    % concat_wav=concat_wav+abs(min(concat_wav));
    concat_wav=concat_wav - mean(concat_wav);
    if maxi>abs(mini)
        up_lim=maxi;
    else
        up_lim=abs(mini);
    end
    concat_wav=concat_wav/up_lim*0.9;
    
    %% Saving
    save_path = [dest_noisy, 'noisy_', num2str(i), '.mat'];
    save(save_path,'concat_wav') %-v7.3 

end
