clear 
close all
clc
%%

% load('rnn_AE_output.mat');
load('conv_AE_output.mat');

%% Data 

y_t_ = y_true_test;
y_p_ = y_pred_test;

% n_data =  length(y_true_test);
% n_data =  n_data - mod(n_data, input_dim);
% y_t_ = y_true_test(1:n_data);
% y_p_ = y_pred_test(1:n_data);
% 
% % y_t_ = reshape(y_t_,[n_data/input_dim, input_dim]);
% % y_p_ = reshape(y_p_,[n_data/input_dim, input_dim]);
% 
% y_t_ = reshape(y_t_,[input_dim, n_data/input_dim])';
% y_p_ = reshape(y_p_,[input_dim, n_data/input_dim])';

%% Hann filtering
% w= hann(input_dim);
% w=w';
% y_p_=filter2(w,y_p_);


%% Trapezoidal filtering to gbet rid of corrupted overlap
% w= trapmf(1:dim*2,[dim/4 dim/2 3/2*dim 7*dim/4]) ;
% y = x.*w; %filter(w, 1, x);



%% fft analysis
y_t_fft = fft (y_t_, size(y_t_,2),2);
y_p_fft = fft (y_p_, size(y_t_,2),2);

% y_t_fft = fft (y_t_);%, input_dim);
% y_p_fft = fft (y_p_);%, input_dim);

% figure
% plot(abs(y_t_fft),'r')
% hold on
% plot(abs(y_p_fft))

figure
plot(abs(mean(y_t_fft,1)),'r')
hold on
plot(abs(mean(y_p_fft,1)))

title('abs(mean(fft))')
xlabel('Sample in the frame (of size 512)')
legend('Original','Reconstructed') 

%% written for the list case
%%%%%%%%%%reshaping just for conv structure
% y=y_true_test;
% y_=y_pred_test;
% 
% y_true_test=zeros(50*64, 2^14);
% y_pred_test=zeros(50*64, 2^14);
% 
% for i=1:50
%     for j=1:64
%         
%         display([i,j])
%         a=y(i,j,:);
%         a=a(:)';
%         a_=y_(i,j,:);
%         a_=a_(:)';
%         
%         y_true_test( (i-1)*64 + j , :) = a;
%         y_pred_test( (i-1)*64 + j , :)  = a_;
%         
%     end
% end

%% Quantization 

% a=fi(y_pred_test);%,1,8);
% b=fi(y_true_test);%,1,8);
% 
% a=quantize(a);
% b=quantize(b);
% 
% y_pred_test=double(a);
% y_true_test=double(b);




%% SNR and Segmented SNR

noise= y_true_test-y_pred_test;
signal_to_noise= 20 * log10( abs(y_true_test./noise));
% signal_to_noise= 20 * log10( max(abs(y_true_test(:))) ./ abs(noise));
signal_to_noise(isinf(signal_to_noise))=[];
signal_to_noise(isnan(signal_to_noise))=[];


snr=signal_to_noise;
snr(snr<0)=0;


snr1 = mean ( snr , 2);
segmented_snr= mean(snr1)

total_snr = mean( snr(:) )



% figure
% plot(snr1)
% title('segmeneted SNR')

%% plotting

figure()

num_fig=16;

for i=1:num_fig
    subplot(sqrt(num_fig),sqrt(num_fig),i)
    ind=randi(size(y_pred_test,1));
    plot(y_pred_test(ind,1:min(512, size(snr,2) ) ) )
    hold on
    plot(y_true_test(ind,1:min(512, size(snr,2) )),'r-')
end


%% soundinG!
num_samples=1 * 1e6;
n_test=size(y_pred_test,1);
sample_true=y_true_test';
sample_true=sample_true(:);
sample_pred=y_pred_test';
sample_pred=sample_pred(:);
 
 
n_data=length(y_t_(:));
rand_ind=randi(n_data-num_samples);
sample_true=sample_true(rand_ind:rand_ind+num_samples);
sample_pred=sample_pred(rand_ind:rand_ind+num_samples);

% n_data=length(y_p_);
% rand_ind=randi(n_data-num_samples);
% sample_true=y_t_(rand_ind:rand_ind+num_samples);
% sample_pred=y_p_(rand_ind:rand_ind+num_samples);




% num_patches=1000;
% n_test=size(y_pred_test,1);
% random_ind=randi(n_test,num_patches,1);
% ind=random_ind:max(n_test, random_ind+num_patches);
% 
% sample_true=reshape(y_true_test(ind,:)',...
%     [1,size(y_true_test,2)*length(ind)]);
% sample_pred=reshape(y_pred_test(ind,:)',...
%     [1,size(y_pred_test,2)*length(ind)]);


% Filtering AE output
% filter_length=10;
% fil = 1/filter_length*ones(filter_length,1);
% sample_pred = filter(fil,1,sample_pred);

%
player = audioplayer(sample_true, 16000, 16);
play(player)   % start the player
pause(5)
stop(player)
display('original finished')


player = audioplayer(sample_pred, 16000, 16);
play(player)   % start the player
pause(5)
stop(player)


%% Writing audio to file


audiowrite('true.wav',sample_true, 16000)
audiowrite('pred.wav',sample_pred, 16000)

