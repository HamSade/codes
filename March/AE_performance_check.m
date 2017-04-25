clear 
close all
clc
%%

load('AE_output.mat');



%% Quantization 

a=fi(y_pred_test);%,1,8);
b=fi(y_true_test);%,1,8);

a=quantize(a);
b=quantize(b);

y_pred_test=double(a);
y_true_test=double(b);




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



figure
plot(snr1)
title('segmeneted SNR')

%%
figure()

num_fig=16;

for i=1:num_fig
    subplot(sqrt(num_fig),sqrt(num_fig),i)
    ind=randi(size(y_pred_test,1));
    plot(y_pred_test(ind,:))
    hold on
    plot(y_true_test(ind,:),'r')
end


%% soundinG!
num_samples=1000000;
n_test=size(y_pred_test,1);
sample_true=y_true_test';
sample_true=sample_true(:);
sample_pred=y_pred_test';
sample_pred=sample_pred(:);


sample_true=sample_true(1:num_samples);
sample_pred=sample_pred(1:num_samples);



% num_patches=1000;
% n_test=size(y_pred_test,1);
% random_ind=randi(n_test,num_patches,1);
% ind=random_ind:max(n_test, random_ind+num_patches);
% 
% sample_true=reshape(y_true_test(ind,:)',...
%     [1,size(y_true_test,2)*length(ind)]);
% sample_pred=reshape(y_pred_test(ind,:)',...
%     [1,size(y_pred_test,2)*length(ind)]);


%% Filtering AE output
% filter_length=10;
% fil = 1/filter_length*ones(filter_length,1);
% sample_pred = filter(fil,1,sample_pred);

%%
player = audioplayer(sample_true, 16000, 16);
play(player)   % start the player
pause(5)
stop(player)

player = audioplayer(sample_pred, 16000);
play(player)   % start the player
pause(5)
stop(player)


