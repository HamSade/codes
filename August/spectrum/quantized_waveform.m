
clc
clear 
close all


load('highway_AE_output.mat')


x = y_true_test';
x = x(:);

x = x/max(abs(x)) *  32;

x = round(x);

plot(x(1000:1000+2048))
