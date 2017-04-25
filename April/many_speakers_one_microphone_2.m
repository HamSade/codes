% function many_speakers_one_microphone(num_speakers)

clear
close all

%% Main parameters

main_path='/local/scratch/PDA/PDAs/';

num_wavs_to_sum=2;
% Lr_rel=rand(1,3);
% H=generate_ir_1r_1m(Lr_rel,num_wavs_to_sum);

num_sums_to_concat=200;
fs_new=16000;

%% Folder and File names

folder_names=dir(main_path);
folder_names=extractfield(folder_names,'name');
folder_names=folder_names(3:end);  % removing . and ..

num_folders=length(folder_names);


%% Body

Lr_rel=rand(1,3);
num_ir=20;
H=generate_ir_1r_1m(Lr_rel, num_ir); %num_wavs_to_sum);

for k=1:num_folders
    
    
    display (k, 'folder index')

    concat_wav=[];

    wav_path=[main_path, '/', folder_names{k}];  
    wav_names=dir(wav_path);
    wav_names=extractfield(wav_names,'name');
    wav_names=wav_names(3:end);
    num_wav_files=length(wav_names);

            
    for i=1:num_sums_to_concat
    
        display(i)
    
        sum_wav=zeros(7e4,1);       
    
        for j=1:num_wavs_to_sum

            wav_ind=randi(num_wav_files);

            [temp_wav, fs]=audioread([wav_path, '/', wav_names{wav_ind}]);
            temp_wav=resample(temp_wav,fs_new, fs);

            % normalizing the signal, bacause I assume they should have same
            % power approximately

            temp_wav=temp_wav/ sum(abs(temp_wav)) * 1e4;

            rand_ind = randi(num_ir, 1, num_wavs_to_sum);
            temp_wav= conv(temp_wav, H(:,rand_ind(j)) );%, 'same');

            if length(temp_wav)<length(sum_wav)
                sum_wav(1:length(temp_wav))=sum_wav(1:length(temp_wav))+temp_wav;
            else
                sum_wav=[sum_wav; zeros( length(temp_wav)-length(sum_wav) ,1)]+temp_wav;

            end
        end


        concat_wav=[concat_wav, sum_wav'];

    end

    concat_wav_raw=concat_wav;


    %% removing noise
    concat_wav=concat_wav_raw;
    concat_wav( abs(concat_wav) > 50*median(abs(concat_wav)))=0;

    %% Scaling signal

    maxi=max(concat_wav);
    mini=min(concat_wav);

    % concat_wav=concat_wav+abs(min(concat_wav));
    concat_wav=concat_wav+mean(concat_wav);

    if maxi>abs(mini)
        up_lim=maxi;
    else
        up_lim=abs(mini);
    end
    concat_wav=concat_wav/up_lim*0.9;

    %% Saving
    save_path = ['/local/scratch/PDA/my_data/com_concat_signal_', num2str(k), '.mat'];
    save(save_path,'concat_wav','fs_new') %-v7.3 

end




