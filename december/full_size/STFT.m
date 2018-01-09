function [stft,t,f] = STFT(x,w, window_size,step_size,fs)
% Copyright 2017: Steven Van Kuyk
% This program comes WITHOUT ANY WARRANTY.
%
% Compute short-time Fourier transform of x (where columns are single-sided
% spectra).

x=x(:);
% w = hann(window_size); % window function (must match window in ISTFT)

frames = 1:step_size:length(x)-window_size;
stft = zeros(window_size,length(frames));
for i=1:length(frames)
    ii = frames(i):frames(i)+window_size-1;
    stft(:,i) = fft(w.*x(ii));
end

% single-sided spectra
stft = stft(1:window_size/2+1,:);
stft(2:end-1,:) = 2*stft(2:end-1,:); 
% stft = stft(1:end-1, :); %remove the last high bin

f = (0:window_size/2)*fs/window_size; % frequency vector
t = (frames-1)/fs; % time vector