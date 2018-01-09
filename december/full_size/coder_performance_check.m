clear 
close all
clc

%%

% load('/vol/grid-solar/sgeusers/hsadeghi/dccoder_results/coder_output_16K.mat')
load('coder_output.mat') 

%%
y_t = double(y_true_test);
y_p = double(y_pred_test);

y_p = cat(2, y_p, y_t(:,end,:));

%%
dim = input_dim;
avg_num = size(maximum_spec,1);
n_batch = size(maximum_spec,2);
fs = 16000;

%% SNR

snr = 0;

for i=1:size(y_t,1)  % for each sample in batch
    
    x = y_t(i, :, :);
    x_hat = y_p(i, :, :);
    
    % mean subtraction
%     x = x - mean(x(:));
%     x_hat = x_hat  -  mean(x_hat(:));
    
    % reverse dB (absolute 512value calc)
    x = 10.^(x/20);
    x_hat = 10.^(x_hat/20);
    
    diff = 10 * log10 (  ( x(:) ./ ( x(:) - x_hat(:) ) ) .^2  );
    diff(isinf(diff))= nan;
    
    snr = snr + nanmean(diff);
end

display(snr/size(y_t,1), 'SNR')


%% Log spectral distortion

lsd_samples = [];
for i=1:size(y_t,1)  
    lsd_local = [];
    for j=1:size(y_t,3)
        
        x1 = y_t(i, :, j);
        x1 = x1 - mean(x1);
        x2 = y_p(i, :, j);
        x2 = x2  -  mean(x2);
        
        diff = (x1 - x2) .^2 ;
        lsd_local = [lsd_local,sqrt(mean(diff))];
    end
    
%     lsd_samples = [lsd_samples, mean(lsd_local)];

    row_ind = floor((i-1)/size(maximum_spec,2)) + 1;
    col_ind = i - (row_ind-1) * size(maximum_spec,2);
    factor = maximum_spec(row_ind, col_ind) / 0.9;
%     factor =1;

    lsd_samples = [lsd_samples, factor * mean(lsd_local)];
end

lsd = 0;
lsd_sub = 0;
spec_window_num_elements = length(lsd_samples)/avg_num;
for i=1:length(lsd_samples)
    
    lsd_sub = lsd_sub + lsd_samples(i);
    if mod(i, spec_window_num_elements) == 0
%         lsd_sub = maximum_spec(i/spec_window_num_elements) / 0.9 * lsd_sub;
        lsd = lsd + lsd_sub;
        lsd_sub = 0;
    end
end

display(lsd/length(lsd_samples), 'LSD')

%% plotting (Tiled fashion)
num_fig=16;

for i=1:num_fig/2
    ind=randi(size(y_t,1));
    
    m = size(y_t,2);
    n = size(y_t,3);
    
    xt_sample = reshape(y_t(ind, :, :), [m,n]);
    xp_sample = reshape(y_p(ind, :, :), [m,n]);  
    
    
    subplot(sqrt(num_fig), sqrt(num_fig),2*(i-1)+1)
%     imshow(xt_sample)
    imagesc((1:size(xt_sample,2))/120, 1:size(xt_sample,1)*32, xt_sample)
    set(gca,'YDir','normal')
    title(['True ', num2str(i)])
    xlabel('sec')
    ylabel('Hz')
    
    subplot(sqrt(num_fig), sqrt(num_fig),2*(i-1) + 2)
%     imshow(xp_sample)
    imagesc((1:size(xp_sample,2))/120, 1:size(xp_sample,1)*32, xp_sample)
    set(gca,'YDir','normal')
    title(['Predicted ', num2str(i)])
    xlabel('sec')
    ylabel('Hz')
end
    

%% SPectrogram to time samples

sample_true= [];
sample_pred = [];

rand_ind = 100;
parfor i=rand_ind:rand_ind+20
    display(i, 'iteration    ')
    
%     coeff = maximum_spec(ceil(i/spec_window_num_elements)) / 0.9;
    row_ind = floor((i-1)/size(maximum_spec,2)) + 1;
    col_ind = i - (row_ind-1) * size(maximum_spec,2);
    coeff = maximum_spec(row_ind, col_ind) / 0.9;
    
%     coeff =1;
    
    xt_sample = coeff * reshape(y_t(i, :, :), [m,n]);
    xp_sample = coeff * reshape(y_p(i, :, :), [m,n]); 
    
    sample_true = [sample_true, griffin_lim(xt_sample)'];
    sample_pred = [sample_pred, griffin_lim(xp_sample)'];
end

%% Sounding

player = audioplayer(sample_true, fs, 16);
play(player)
pause(10)
stop(player)

display('original finished')

player = audioplayer(sample_pred, fs, 16);
play(player)   % start the player
pause(10)
stop(player)

%% Writing audio to file

audiowrite('true.wav',sample_true, fs)
audiowrite('pred.wav',sample_pred, fs)

