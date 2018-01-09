function x_final = griffin_lim(Y)

%%

% dB to linear 
Y = 10.^(Y/20);
% display(size(Y), 'size(Y)')
% Y = [Y; zeros(1, size(Y,2))];


num_iters = 100;
fs = 16000;
nperseg = floor( 32 /1000 * fs);  % 50 ms = 800 samples
overlap = floor(nperseg * 0.75);
S = nperseg - overlap;

%% Window
% w = tukeywin(nperseg, 0.25);
% w= 1/sqrt(2)*sqrt(hann(nperseg));
% w = sqrt(S /nperseg)*ones(nperseg,1); 
% w = sine_window(S,nperseg)';
% w = sqrt(hann(nperseg));
w = hann(nperseg);

%% ALGO
% x_istft = randn([1, input_dim]).*exp(1i*2*pi*rand([1, input_dim]));
% X = STFT(x_istft,w, nperseg,S,fs);
% X = [X , zeros([size(X,1),5])];

X = exp(2*pi*1i*rand(size(Y))); 
% display(size(X), 'size(X)')

for k=1:num_iters
    x_istft = ISTFT(Y.*exp(1j*angle(X)),w,nperseg,S);
%     display(size(x_istft), 'x_istft')
    X = STFT(x_istft,w, nperseg,S,fs);
%     display(size(X), 'X')
end

%% Plots
% figure(1)
% imagesc(log(abs(stft)))
% set(gca,'ydir','normal')
% title('target magnitude spectrum')
% 
% figure(2)
% plot((1:length(x))/fs,x, (1:length(x_istft))/fs,x_istft,':')
% legend('original signal', 'synthesized signal')

x_final = x_istft;


%% Sounding
% sound([x; x_final],fs) 
