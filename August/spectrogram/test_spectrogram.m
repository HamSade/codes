
clc
clear

%%
% x = rand([1, 256 *10]);
fs = 10e3;
t = 0:1/fs:2;
x = vco(sawtooth(2*pi*t,0.5),[0.1 0.4]*fs,fs);

noverlap =  0;
nfft = 1024;
window = 256;

spectrogram(x,kaiser(256,5),220,512,fs,'yaxis')

% spectrogram(x,window,noverlap,nfft, 'yaxis');