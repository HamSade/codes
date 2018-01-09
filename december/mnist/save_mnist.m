clc
clear
close all


%%

load_path = '/vol/grid-solar/sgeusers/hsadeghi/MNIST/';

%%

images = loadMNISTImages([load_path,'train-images-idx3-ubyte']);
labels = loadMNISTLabels([load_path, 'train-labels-idx1-ubyte']);

%%
database=zeros(60000, 32, 32);

for i=1:60000
    i
    I = images(:,i);
    I =reshape(I, [28, 28]);
    pad = zeros(32);
    pad(3:30, 3:30) = I;
    database(i, :, :) = pad;
end
    
    
%%
save_path = '/vol/grid-solar/sgeusers/hsadeghi/MNIST/mat_mnist/';
save_filename = [save_path, 'database.mat'];

save(save_filename, 'database', 'labels');

%%
% imshow(I)
% title(num2str(labels(1)))