clear 
close all
clc
%%

load('highway_AE_output.mat') 

%%
% y_t_raw = y_true_test;
% y_p_raw = y_pred_test;

y_t_raw = y_h_true;
y_p_raw = y_h_pred;

dim = input_dim;
fs = 16000;

%% copying raw data

y_t = y_t_raw;
y_p = y_p_raw;


%% LSD (Log-Spectral distortion)

% dB computation
y_p = abs(y_p);
y_t = 20*log10(y_t);
y_p = 20*log10(y_p);


% Making all rows zero-mean
y_t = y_t -  mean(y_t, 2);
y_p = y_p -  mean(y_p, 2);

% Calculating lsd like RMS of columns averaged along time (columns)
lsd =  nanmean ( sqrt(  nanmean ( (y_t - y_p).^2 , 2) ) )

%% plotting
num_fig=36;
figure()

for i=1:num_fig  
    ind=randi(size(y_t_raw,1));
%     xt_sample = y_t_raw(ind, :);
%     xp_sample = y_p_raw(ind, :);
    xt_sample = y_t(ind, :);
    xp_sample = y_p(ind, :);
    
    subplot(sqrt(num_fig),sqrt(num_fig),i)
    plot(xt_sample, 'r')
    hold on
    plot(xp_sample)
end

%% Spectrum to time domain conversion


%% soundinG!
% num_samples=1 * 1e6;
% 
% n_data=length(y_p_);
% rand_ind=randi(n_data-num_samples);
% sample_true=y_t_raw(rand_ind+1:rand_ind+num_samples);
% sample_pred=y_p_raw(rand_ind+1:rand_ind+num_samples);
% 
% player = audioplayer(sample_true, fs, 16);
% play(player)   % start the player
% pause(5)
% stop(player)
% display('original finished')
% 
% player = audioplayer(sample_pred, fs, 16);
% play(player)   % start the player
% pause(10)
% stop(player)
% 
% %% Writing audio to file
% audiowrite('true.wav',sample_true, fs)
% audiowrite('pred.wav',sample_pred, fs)

