
% Generating many rooms impulse responses

%% Parametrs

c = 340;                    % Sound velocity (m/s)
fs = 16000;                 % Sample frequency (samples/s)
n_samples = 4096;                   % Number of samples

% Room dimension ranges
x_min=2;
x_max=6;
z_min=2.5;
z_max=4;

% other parameters
mtype_options=['omnidirectional', 'subcardioid',...
'cardioid', 'hypercardioid', 'bidirectional']; % microphone types


%% Random room impulse response generations 

num_simulations=1000;
H=zeros(n_samples,num_simulations);

parfor i=1:num_simulations
    
    if mod(i,100)==0
        disp(i);
    end
    
    
    L = [(x_max-x_min)*rand+x_min...
        (x_max-x_min)*rand+x_min...
        (z_max-z_min)*rand+z_min];                % Room dimensions [x y z] (m)
    
    r = [L(1)*rand L(2)*rand L(3)*rand];          % Receiver position [x y z] (m)
    s = [L(1)*rand L(2)*rand L(3)*rand];          % Source position [x y z] (m)


    % reflection coefficient determination
    %     % option 1
%     % Calculation beta from RT_60= 24 ln(10) * V/sig
%     sig=24*log(10)*120/0.4; % 120 and 0.4 come from examples
%     beta = 0.4;                 % Reverberation time (s)
    
    % option 2
    beta_x=rand;
    beta_floor=rand;
    beta_ceiling=rand;
    
    beta=[beta_x beta_x beta_x beta_x beta_floor beta_ceiling];
    
    % Mic type selection
    type_ind=randi(5);
    switch type_ind
        case 1
            orientation=0;
            order = [];
        otherwise
            orientation = [2*pi*rand -pi/2+pi*rand];     % Microphone orientation (rad)  
            order=-1;
    end
            
            
            
    mtype = mtype_options(type_ind);    % Type of microphone
                     % -1 equals maximum reflection order!
    dim = 3;                    % Room dimension
    
    hp_filter = randi(2)-1;              % Enable/Disable high-pass filter


    H (:,i)= rir_generator(c, fs, r, s, L, beta, n_samples, mtype, order, dim, orientation, hp_filter);
    
    
end


%% Saving results

save impulse_responses H fs n_samples x_min x_max z_min z_max -v7.3
 



