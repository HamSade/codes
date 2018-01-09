function istft = ISTFT(stft,w, window_size,step_size)
% Copyright 2017: Steven Van Kuyk
% This program comes WITHOUT ANY WARRANTY.
%
% Compute inverse short-time Fourier transform of stft (where columns are
% single-sided spectra).

% w = hann(window_size); % window function (must match window in STFT)

% two-sided spectra
stft(2:end,:) = stft(2:end,:)/2;
stft = [stft; flipud(conj(stft(2:end-1,:)))]; 
% stft = [stft; zeros(2, size(stft,2)); flipud(conj(stft(2:end-1,:)))];
% zeros(2, size(stft,2))  to compensate for the removed high bin

% apply algorithm from Griffin & Lim 1984 (similar to overlap add)
istft = zeros(size(stft,2)*step_size+window_size,1);
w2 = zeros(size(istft));
frames = 1:step_size:length(istft)-window_size;
for i=1:length(frames)
    ii = frames(i):frames(i)+window_size-1;
    istft(ii) = istft(ii)+w.*ifft(stft(:,i)); % equation 6 numerator
    w2(ii) = w2(ii) + w.^2; % equation 6 denominator
end
w2(w2<0.001) = 0.001; % for stability

istft = istft./w2; % equation 6

istft = real(istft);