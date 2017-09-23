clear 
close all
clc
%%
load('gan_dc_spec_gen_output.mat') 

%%
y_t_raw = y_true_test;
y_p_raw = y_pred_test;
dim = input_dim;
n_batch = size(y_t_raw,1)/50;
fs = 16000;

%% copying raw data

y_t = y_t_raw;
y_p = y_p_raw;


%% LSD (Log-Spectral distortion)


% lsd = zeros(1, size(y_t,1));
% 
% for i=1:size(y_t,1)
%     
%     maxi = maximum_spec(ceil(i/n_batch));
%     
%     % Making all columns zero-mean
%     y_t(i, :, :) = y_t(i, :, :) -  mean(y_t(i, :, :)) ;%* maxi;
%     y_p(i, :, :) = y_p(i, :, :) -  mean(y_p(i, :, :)) ;%* maxi;
%     
%     % Calculating lsd like RMS of columns averaged along time (columns)
%     lsd(i) =  mean  ( sqrt(  mean ((maxi* (y_t(i, :, :) - y_p(i, :, :)) ).^2 ) ) );
% end
% 
% lsd =  mean(lsd)

%% plotting concatenated

num_fig=8;


xt_sample = [];
xp_sample = [];

for i=1:num_fig*2   
    ind=randi(size(y_t_raw,1));
    xt_sample = [xt_sample squeeze(y_t_raw(ind, :, :))];
    xp_sample = [xp_sample squeeze(y_p_raw(ind, :, : ))];
    
end

figure()
subplot(2,1,1)
imagesc(1:size(xt_sample,2), 1:size(xt_sample,1), xt_sample)

subplot(2,1,2)
imagesc(1:size(xp_sample,2), 1:size(xp_sample,1), xp_sample)

%% plotting side by side
% 
% num_fig=4;
% figure()
% 
% for i=1:2:num_fig*2   
%     ind=randi(size(y_t_raw,1));
%     xt_sample = squeeze(y_t_raw(ind, :, :));
%     xp_sample = squeeze(y_p_raw(ind, :, : ));
%     
%     subplot(num_fig,2,i)
%     imagesc(1:size(xt_sample,2), 1:size(xt_sample,1), xt_sample)
%     
%     subplot(num_fig,2,i+1)
%     imagesc(1:size(xt_sample,2), 1:size(xt_sample,1), xp_sample)
% end

%% Spectrogram to time domain conversion


%% soundinG!
% num_samples=1 * 1e6;
% 
% n_data=length(y_p_);
% rand_ind=randi(n_data-num_samples);
% sample_true=y_t_(rand_ind+1:rand_ind+num_samples);
% sample_pred=y_p_(rand_ind+1:rand_ind+num_samples);
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

