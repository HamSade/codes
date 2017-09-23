clear 
close all
clc
%%

% load('rnn_AE_output.mat');
% load('GAN_output.mat');
% load('L1_AE_output.mat')
load('highway_AE_output.mat')
% load('GAN_highway_output.mat')


y_t = y_true_test;
y_p = y_pred_test;
dim = input_dim;

fs = 16000

%% Plotting spectrogram

% s_t = spectrogram(y_t(:));
% s_p = spectrogram(y_p(:));
% s_l = spectrogram(y_t(:)-y_h_true(:));

% spectrogram(y_t(:), 'yaxis');
% figure()
% spectrogram(y_p(:), 'yaxis');
% figure()
% spectrogram(y_t(:)-y_h_true(:), 'yaxis');
% 

%% Huffman coding

% % codes = codes';
% codes = codes(:);
% 
% exisiting_codes = unique(codes)'
% 
% symbols = [-1, 0, 1];
% len_codes =  sum(codes~=0);
% 
% freq_pos = sum(codes==1);
% p_pos = freq_pos / len_codes;
% freq_neg = sum(codes==-1);
% p_neg = freq_neg / len_codes;
% 
% dict = huffmandict(symbols,[p_neg, 0, p_pos]);
% comp_codes = huffmanenco(codes,dict);
% comp_ratio = length(codes) / length(comp_codes)

%% Small noise canceller filter
% n_filt=2;
% w= 1/n_filt*ones(1,n_filt);
% y_p=filter2(w,y_p);
% % y_p= medfilt1(y_p',n_filt)';

%% Hann filtering
% w= hann(input_dim);
% w=w';
% y_p=filter2(w,y_p);

%% Filtering to compensate for rect window  
y_t_ = zeros(1,size(y_t,1) * dim); % batch_size x input_dim
y_p_ = zeros(1,size(y_t,1) * dim);

for i=1:size(y_t,1)
    
    data_window = y_t (i, overlap+1:overlap+dim);  
    ind_range = 1+ (i-1)*(dim) : i*(dim);
    
    y_t_(ind_range)= y_t_(ind_range) + data_window ;    
    
    data_window = y_p (i, overlap+1:overlap+dim);
    y_p_(ind_range)= y_p_(ind_range) + data_window ;
end

%% Filtering to compensate for rect window just for the highpass signals 
y_h_t = zeros(1,size(y_h_true,1) * dim); % batch_size x input_dim
y_h_p = zeros(1,size(y_h_true,1) * dim);

for i=1:size(y_h_t,1)
    
    data_window = y_h_true (i, overlap+1:overlap+dim);  
    ind_range = 1+ (i-1)*(dim) : i*(dim);
    
    y_h_t(ind_range)= y_h_t(ind_range) + data_window ;    
    
    data_window = y_h_pred (i, overlap+1:overlap+dim);
    y_h_p(ind_range)= y_h_p(ind_range) + data_window ;
end
%% Filtering to compensate for the Trapezoidal window
%  
% y_t_ = zeros(1,overlap + size(y_t,1) * (dim+overlap/2)); % batch_size x input_dim
% y_p_ = zeros(1,overlap + size(y_t,1) * (dim+overlap/2));
% % w= trapmf(1:dim*2,[dim/4 dim/2 3/2*dim 7*dim/4]) ;
% 
% % ind_range_initial = overlap/2+1:dim+overlap/2;
% ind_range_initial = 1: dim + overlap * 2;
% y_t_(ind_range_initial) = y_t (1, ind_range_initial);
% y_p_(ind_range_initial) = y_p (1, ind_range_initial);
% 
% for i=2:size(y_t,1)-1
%     data_window = y_t (i, :);
%     
%     ind_range = 1+ (i-1)*(dim+ overlap/2) : i*(dim + overlap/2)+3*overlap/2;
%    
%     y_t_(ind_range)= y_t_(ind_range) + data_window ;    
%     data_window = y_p (i, :);
%     y_p_(ind_range)= y_p_(ind_range) + data_window ;
% end

%% fft analysis
% y_t_fft = fft (y_t_, size(y_t_,2),2);
% y_p_fft = fft (y_p_, size(y_t_,2),2);
% % y_t_fft = fft (y_t_);%, input_dim);
% % y_p_fft = fft (y_p_);%, input_dim); 
% % figure
% % plot(abs(y_t_fft),'r')
% % hold on
% % plot(abs(y_p_fft))
% figure
% plot(abs(mean(y_t_fft,1)),'r')
% hold on
% plot(abs(mean(y_p_fft,1))) 
% title('abs(mean(fft))')
% xlabel('Sample in the frame (of size 512)')
% legend('Original','Reconstructed') 

%% SNR and Segmented SNR

% noise= y_t_-y_p_;
% signal_to_noise= 20 * log10( abs(y_t_./noise));

noise= y_h_t-y_h_p;
signal_to_noise= 20 * log10( abs(y_h_t./noise));

signal_to_noise(isinf(signal_to_noise))=[];
signal_to_noise(isnan(signal_to_noise))=[];
snr=signal_to_noise;
snr(snr<0)=0;
snr1 = mean ( snr , 2);
segmented_snr= mean(snr1)
total_snr = mean( snr(:) )

%% plotting

figure()
num_fig=16;
for i=1:num_fig
    subplot(sqrt(num_fig),sqrt(num_fig),i)
    ind=randi(size(y_pred_test,1)-1);
    plot(y_p_(ind*dim+1:(ind+1)*dim))
    hold on
    plot(y_t_(ind*dim+1:(ind+1)*dim),'r-')
end

%% soundinG!
num_samples=1 * 1e6;

n_data=length(y_p_);
rand_ind=randi(n_data-num_samples);
sample_true=y_t_(rand_ind+1:rand_ind+num_samples);
sample_pred=y_p_(rand_ind+1:rand_ind+num_samples);

player = audioplayer(sample_true, fs, 16);
play(player)   % start the player
pause(5)
stop(player)
display('original finished')

player = audioplayer(sample_pred, fs, 16);
play(player)   % start the player
pause(10)
stop(player)

%% Writing audio to file
audiowrite('true.wav',sample_true, fs)
audiowrite('pred.wav',sample_pred, fs)

