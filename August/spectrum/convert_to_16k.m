clear
close all


%%

path_name = '/home/hsadeghi/Downloads/segan/test_results/noisy/';
[y,fs] = audioread([path_name, 'altouna.wav']);

fs

%%
fs_new= 16000;
y_new = resample(y,fs_new,fs);

% y_new = y_new(1:2*fs_new);
%%

player = audioplayer(y_new, fs_new, 16);
play(player)   % start the player
pause(3)
stop(player)

%%
% audiowrite('/home/hsadeghi/Downloads/segan/test_results/noisy/altouna_16k.wav',y_new, fs_new)