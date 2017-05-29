%%%

clear
close all


%%
main_path='/local/scratch/PDA/PDAs/';
fs_new=16000;

folder_names=dir(main_path);
folder_names=extractfield(folder_names,'name');
folder_names=folder_names(3:end);  % removing . and ..

wav_path=[main_path, '/', folder_names{1}];  
wav_names=dir(wav_path);
wav_names=extractfield(wav_names,'name');
wav_names=wav_names(3:end);
num_wav_files=length(wav_names);

%%

% wav_ind=randi(num_wav_files);
wav_ind = 1;

file_name =  [wav_path, '/', wav_names{wav_ind}];

[temp_wav, fs]=audioread(file_name, [1,50000]); %, 'native');

output = resample(double(temp_wav), fs_new, fs);


output_16 = int16( output * (2^16-1));
output_8  = int8(output * (2^8-1) );


%%

sum(abs( double(output_16) / (2^16-1)  -double(output_8) / (2^8-1) )) / length(output)

%%
% histogram(output*scale_factor, 2^16);
% figure
% histogram(output_, 2^8);

%%
% player = audioplayer(output, fs_new, 16);
% play(player)   % start the player
% pause(5)
% stop(player)










