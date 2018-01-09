clc
clear
close all

%%
% path_name = '/vol/grid-solar/sgeusers/hsadeghi/data/';
path_name = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/mat_clean_16k/';
% path_name = '/vol/grid-solar/sgeusers/hsadeghi/simulated_data/poisson_pulse_train/'

%%

fs = 16000;
nperseg = floor( 32 /1000 * fs );
noverlap = nperseg * 3 / 4;

save_path = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/mat_spec_clean_16k/';

%%

% file_ind = 0;
for file_ind = 0:10
     
    file_ind
    
    load([path_name, 'clean_', num2str(file_ind), '.mat'])
    % concat_wav = [1, -1]
    
    window = hann(nperseg);
    Sxx = abs( spectrogram(concat_wav,window,noverlap) ); 
    
%     Sxx = Sxx(1:end-1,:);
    Sxx = 20. * log10(Sxx); 
    
%     Sxx = Sxx - mean(Sxx);

    save_filename = [save_path , 'clean_spec_', num2str(file_ind), '.mat'];
    save(save_filename, 'Sxx', 'nperseg', 'noverlap');

end


