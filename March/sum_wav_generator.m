
% function sum_wav=sum_wav_generator(num_wav_to_sum,fs_new)

clear
close all


%% Main parameters

main_path='/home/hsadeghi/Downloads/PDA/PDAs';

IR = matfile('impulse_responses.mat') ;%/home/hsadeghi/Downlo…/RIR-Generator-master/
num_IR=size(IR.H,2);

num_wav_to_sum=5;
fs_new=16000;

%% body
folder_names=dir(main_path);
folder_names=extractfield(folder_names,'name');
folder_names=folder_names(3:end);  % removing . and ..

num_folders=length(folder_names);


sum_wav=zeros(7e4,1);
% concat_wav=[];


for i=1:num_wav_to_sum
    
    folder_ind=randi(num_folders);
    wav_path=[main_path, '/', folder_names{folder_ind}];
    
    
    wav_names=dir(wav_path);
    wav_names=extractfield(wav_names,'name');
    wav_names=wav_names(3:end);
    
    num_wav_files=length(wav_names);
    wav_ind=randi(num_wav_files);
    
    [temp_wav, fs]=audioread([wav_path, '/', wav_names{wav_ind}]);
    temp_wav=resample(temp_wav,fs_new, fs);
    
    % normalizing the signal, bacause I assume they might be of the same
    % power approximately

    temp_wav=temp_wav/ sum(abs(temp_wav)) * 1e4;
   
    
    
    % Select and apply room IR
    ind_IR=randi(num_IR);
    selected_IR=IR.H(:, ind_IR);
        
    temp_wav= conv(temp_wav, selected_IR);%, 'same');
    
    if length(temp_wav)<length(sum_wav)
        sum_wav(1:length(temp_wav))=sum_wav(1:length(temp_wav))+temp_wav;
    else
        sum_wav=[sum_wav; zeros( length(temp_wav)-length(sum_wav) ,1)]+temp_wav;
        
    end
    
%     concat_wav=[concat_wav, temp_wav'];
    
end



plot(sum_wav)

player = audioplayer(sum_wav, fs_new);
play(player)   % start the player
% stop(player)


%% Saving
% save 






