
clc
clear 

%%  Toderici
 
data = 216* (10^6)* 32 * 32 * 3*8;

% network
x=3072;

inenc= x * 512+ 512;

codec=4 * (512*512 + 512);

mid= 512 * 128 + 128;

indec= 128* 512+ 512;

out= 512 * x + x;


params_per_step = inenc+codec+mid+indec+out;
params = 16 * params_per_step;

% dpp
dpp= data/params




%% ours

data = 32772992 * 16;

% network
x=128;


full=1024;
binary=20;
num_steps=2;
num_layers=0



inenc= x * full+ full;
codec=num_layers * (full*full + full);
mid= full * binary + binary;
indec= 16* full+ full;
out= full * x + x;
params = num_steps * (inenc+codec+mid+indec+out);

dpp= data/params












