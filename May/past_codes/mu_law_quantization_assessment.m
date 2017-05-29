

%% 
 clear
 close all
 
%% 
% path_name = '/home/hsadeghi/Downloads/Data/';
% file_name = [path_name 'com_concat_signal_1.mat'];
% load(file_name);

load('data.mat')

%% Mu-law compression

mu= 255;
n_bits=8;
fs=16000;

% x = concat_wav;
% 
% x= x(1:100000 );

x_compressed = compand(x,mu,0.9,'mu/compressor');  % The maximum of amplitude is garaunteed to be 0.9

ntBP = numerictype( 1, n_bits, n_bits -1 );  % n-bit representation

x_fi = fi(x_compressed, ntBP);

x_quantized =  quantize(x_fi, 1, n_bits);  % The output is signed and uses n_bit bits

x_double =  double(x_quantized);

x_companded = compand(x_double,mu,0.9,'mu/expander');


%%

player = audioplayer(x , fs, 16);
play(player)   % start the player
pause(6.5)
stop(player)

disp('original completed')

player = audioplayer(x_companded, fs, n_bits);
play(player)   % start the player
pause(6.5)
stop(player)

%%
audiowrite('16_bit_speech.wav', x, fs,'BitsPerSample',16)
audiowrite('8_bit_speech.wav', x_companded, fs,'BitsPerSample',n_bits)
save data.mat x

