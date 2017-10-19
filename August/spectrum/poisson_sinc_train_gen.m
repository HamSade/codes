function [filtered_speech] = poisson_sinc_train_gen(block_size, fs, fc)

apply_plot=1;

if nargin == 0 
    fs = 16000;
    fc = 8000; %Filter cutoff frequency
    block_size = 512;
end
min_time = 5e-3;
resolution = 1 / fs ;

poiss_rndm_intrvls = [];
while isempty(poiss_rndm_intrvls)
    poiss_rndm_intrvls = exprnd(min_time,1,block_size)/resolution;
end

poiss_times = cumsum(poiss_rndm_intrvls);
poiss_times(poiss_times > block_size) =[];
% plot(poiss_times)

speech_block = zeros(1, block_size);

if ~isempty(poiss_times)
    speech_block(ceil(poiss_times))=1;
end

%%
block_times = (0: block_size-1)*resolution;
filt_ = 0.5 * sinc(2*pi*fc* block_times);

filtered_speech = conv(speech_block, filt_); %(end:-1:1));
filtered_speech = filtered_speech(1:block_size);


%% Plotting
if apply_plot == 1
    speech_fft =  fft(filtered_speech);    
    plot(abs(speech_fft))
end
 
% if apply_plot==1
%     stem(speech_block)
%     figure
%     plot(filt_);
%     figure
%     plot(filtered_speech);
% end










































