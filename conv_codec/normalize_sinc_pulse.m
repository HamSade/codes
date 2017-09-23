for i=0:10
    
    
    display( i )
    load_path = ['/vol/grid-solar/sgeusers/hsadeghi/simulated_data/poisson_pulse_train/data_',num2str(i),'.mat'];

    load(load_path)
    
    x =  concat_wav;
    maxi = max(x);
    mini = min(x);
    
    A = inv([maxi, 1; mini,1 ])*[1;-1];
    a = A(1);
    b =  A(2);
    x= a*x + b;   

    concat_wav =  x;
    save_path = ['/vol/grid-solar/sgeusers/hsadeghi/simulated_data/poisson_pulse_train_normalized/data_',num2str(i),'.mat'];
%     save save_path concat_wav


end


%%
% clc
% clear
% close all

%
% % inv([0.5 1; -0.05 1])*[1;-1]
% % 
% % ans =
% % 
% %     3.6364
% %    -0.8182
% 
% 
% load_path = ['/vol/grid-solar/sgeusers/hsadeghi/simulated_data/poisson_pulse_train/data_',num2str(0),'.mat'];
% 
% load(load_path)
%     
% x = concat_wav;
% figure
% plot(x)
% 
% maxi = max(x);
% mini = min(x);
% A = inv([maxi, 1; mini,1 ])*[1;-1];
% a = A(1);
% b =  A(2);
% x= a*x + b;
% 
% concat_wav = x;
% figure
% plot(x(100:1000))











