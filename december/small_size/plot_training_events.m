clc
clear 
close all

%%

load_path = '/vol/grid-solar/sgeusers/hsadeghi/research_results/saved_model';
load([load_path,'/events_training.mat']) 

%%
figure()
plot(coder_cost)
title('coder_{cost}')

figure()
plot(disc_cost)
title('disc_{cost}')

% figure()
% x = l1_cost;
% % x = filter(ones(5,1),1,x);
% plot(x)
% title('L1_{cost}')
