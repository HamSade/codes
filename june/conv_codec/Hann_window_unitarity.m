close all;
clear

%%
dim = 512;
x= 5 * rand(1,dim*2);

% x= ones(1,dim);


%% Hann window 

% w= hann(dim);
% % y = filter(w, 1, x);
% y = conv(x, w);
% x_ = conv(y, w);
% figure()
% plot(x)
% hold on
% plot(x_)

%% Trapezoid filter

w= trapmf(1:dim*2,[dim/4 dim/2 3/2*dim 7*dim/4]) ;
% y = filter(w, 1, x);
y= x.*w;

figure()
plot(w)

figure()
plot(x)
hold on
plot(y,'r')
