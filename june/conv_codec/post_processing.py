#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 14:35:28 2017

@author: hsadeghi
"""

import numpy as np
import scipy
import scipy.io as sio
from numpy.fft import fft, fftshift

import matplotlib.pyplot as plt

#%% Testing

file_path = "/home/hsadeghi/Dropbox/june/conv_codec/conv_AE_output.mat"

play_audio(file_path)

#%%
def play_audio(file_path=file_path):
    
    #%%
    loaded_data = sio.loadmat(file_path)
    y_true_test = loaded_data['y_true_test']
    y_pred_test = loaded_data['y_pred_test'] 
    
    #%%
    input_dim = y_pred_test.shape[1]
    y_t_ = y_true_test;
    y_p_ = y_pred_test;
        
    #%% Hann filtering
    w= np.hanning(input_dim);
    # filter in scipy reverses w, but since our w is symmetric, we don't need to reverse its order
    y_p_ = scipy.signal.lfilter(w,1, y_p_, axis=-1)

    #%% fft analysis
#    y_t_fft = fft (y_t_, size(y_t_,2),2);
#    y_p_fft = fft (y_p_, size(y_t_,2),2);
#    
#    % y_t_fft = fft (y_t_);%, input_dim);
#    % y_p_fft = fft (y_p_);%, input_dim);
#    
#    % figure
#    % plot(abs(y_t_fft),'r')
#    % hold on
#    % plot(abs(y_p_fft))
#    
#    figure
#    plot(abs(mean(y_t_fft,1)),'r')
#    hold on
#    plot(abs(mean(y_p_fft,1)))
#    
#    title('abs(mean(fft))')
#    xlabel('Sample in the frame (of size 512)')
#    legend('Original','Reconstructed') 
    
    #%% SNR and Segmented SNR
    
    noise = np.subtract( y_true_test, y_pred_test);
    signal_to_noise= 20 * np.log10( np.abs( np.divide( y_true_test , noise )))
    
#    signal_to_noise(np.isinf(signal_to_noise))=[];
#    signal_to_noise(np.isnan(signal_to_noise))=[];
  
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
    
    
    #%% soundinG!
    
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
    pause(10)
    stop(player)
    display('original finished')
    
    
    player = audioplayer(sample_pred, 16000, 16);
    play(player)   % start the player
    pause(10)
    stop(player)
    
    
    #%% Writing audio to file

    audiowrite('true.wav',sample_true, 16000)
    audiowrite('pred.wav',sample_pred, 16000)

