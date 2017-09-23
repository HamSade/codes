

clc
clear 
close all


%%

block_size=512;
fs = 16000;
fc = 8000;

concat_wav=[];
n = 9;

for i=1: n
    if mod(i,100)==0
        display(i, 'i')
    end
    
    concat_wav=[concat_wav; poisson_sinc_train_gen(block_size, fs, fc)];
end
%%
figure
for i=1:n
    subplot(sqrt(n),sqrt(n),i)
    plot(concat_wav(i,:))
end

figure
for i=1:n
    subplot(sqrt(n),sqrt(n),i)
    plot(abs(fft(concat_wav(i,:))))
end


%%
% player = audioplayer(concat_wav, fs, 16);
% play(player)   % start the player
% pause(5)
% stop(player)


%%
% save_path = '/vol/grid-solar/sgeusers/hsadeghi/simulated_data/poisson_pulse_train/data_1.mat';

% save /vol/grid-solar/sgeusers/hsadeghi/simulated_data/poisson_pulse_train/data_10.mat concat_wav








