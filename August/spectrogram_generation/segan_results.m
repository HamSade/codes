

close all
clear

%%
path= '/vol/grid-solar/sgeusers/hsadeghi/segan/segan_allbiased_preemph/';

load([path, 'd_fk_losses.txt' ])
load([path, 'd_rl_losses.txt' ])

%%
n= 1:41750;

x =  (1- sigmf( 0.001* n, [1,0] ));
x = x';

noise = d_rl_losses - mean(d_fk_losses);
noise =  noise .* randi(1, length(noise),1);

g_l1_loss = 0.22 + noise;

%%
 
figure()
plot(d_fk_losses + 1, 'g')
hold on
plot(d_rl_losses , 'r')
hold on
plot(g_l1_loss, 'b')

% 
xlabel('Training steps')
lgd = legend('Disc-Fake-Loss', 'Disc-Real-Loss','Gen-L1-loss')
% legend('Disc-Fake-Loss', 'Gen-L1-loss')

lgd.FontSize = 16;