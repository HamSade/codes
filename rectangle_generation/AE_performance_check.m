clear 
close all
clc
%%
% load('dcgan_spectrogram_output_32k_wasserstein.mat') 
load('dcgan_spectrogram_output.mat')
% load('disc_first_output.mat')

%%
y_t_raw = y_true_test;
y_p_raw = y_pred_test;
dim = input_dim;
n_batch = size(y_t_raw,1)/50;
fs = 16000;

%% copying raw data

y_t = y_t_raw;
y_p = y_p_raw;


%% 2D Gaussian filtering


filtering = 0;

if filtering ~=0
    
    h_g = fspecial('gaussian', 5, 5);

    for i=1:size(y_t,1)
        y_temp = squeeze(y_t(i,:,:));
        y_temp = filter2(h_g,  y_temp);
        % hole insertion
        hole_mask = floor(randi(2, size(y_temp))/2);
        y_temp = y_temp .* hole_mask;

        y_t(i,:,:) = y_temp;
    end

end
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


%% plotting (Tiled fashion)

num_fig=100;

for i=1:num_fig/2
    ind=randi(size(y_t_raw,1));
    
    n = size(y_t,2);
    
    xt_sample = reshape(y_t(ind, :, :), [n,n]);
    xp_sample = reshape(y_p(ind, :, :), [n,n]);  
    
    
    subplot(sqrt(num_fig), sqrt(num_fig),i)
%     imshow(xt_sample)
    imagesc(1:size(xt_sample,2), 1:size(xt_sample,1), xt_sample)
    
    subplot(sqrt(num_fig), sqrt(num_fig),i + num_fig/2)
%     imshow(xp_sample)
    imagesc(1:size(xt_sample,2), 1:size(xt_sample,1), xp_sample)
end




 %% plotting concatenated
% 
% num_fig=36;
% 
% 
% xt_sample = [];
% xp_sample = [];
% 
% for i=1:num_fig * 2   
%     ind=randi(size(y_t_raw,1));
%     
%     n = size(y_t,2);
%     
%     xt_sample = [xt_sample reshape(y_t(ind, :, :), [n,n])];
%     xp_sample = [xp_sample reshape(y_p(ind, :, :), [n,n])];  
% % %     xt_sample = [xt_sample squeeze(y_t(ind, :, :))];
% % %     xp_sample = [xp_sample squeeze(y_p(ind, :, : ))];
%     
% %     x_sample = [xt_sample xp_sample];
%     
% %     subplot(sqrt(num_fig), sqrt(num_fig), i)
% %     imshow(x_sample)
%     
%     
% end
% 
% 
% figure()
% subplot(2,1,1)
% imshow(xt_sample)
% % imagesc(1:size(xt_sample,2), 1:size(xt_sample,1), xt_sample)
% 
% subplot(2,1,2)
% imshow(xp_sample)
% % imagesc(1:size(xp_sample,2), 1:size(xp_sample,1), xp_sample)

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

