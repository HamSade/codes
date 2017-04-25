
% Generating many rooms impulse responses

function H=generate_ir_1r_1m(Lr_rel,num_ir)

%%
% Lr_relative should be sth like [rand rand rand]

if nargin==0
    Lr_rel=rand(1,3);
    num_ir=5;
end

%% Parametrs

c = 340;                    % Sound velocity (m/s)
fs = 16000;                 % Sample frequency (samples/s)
n_samples = 4096;                   % Number of samples

% Room dimensions
L =[6 3.7 2.8]; % Room dimensions [x y z] (m)


% other parameters
% orientation = [2*pi*rand -pi/2+pi*rand];
orientation=0;

mtype='omnidirectional';
% , 'subcardioid','cardioid', 'hypercardioid', 'bidirectional']; % microphone types
order=-1;
hp_filter = randi(2)-1;              % Enable/Disable high-pass filter


r = Lr_rel.*L;        % Receiver position [x y z] (m)

% reflection coefficient determination
beta_x=rand;
beta_floor=rand;
beta_ceiling=rand;
beta=[beta_x beta_x beta_x beta_x beta_floor beta_ceiling];  
dim = 3;                    % Room dimension

%% Random room impulse responses generation

H=zeros(n_samples,num_ir);

for i=1:num_ir

    s = [L(1)*rand L(2)*rand L(3)*rand];           % Random source position [x y z] (m)
    H(:,i)= rir_generator(c, fs, r, s, L, beta, n_samples, mtype,...
        order, dim, orientation, hp_filter);

end
