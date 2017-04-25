% Analysis of AE distribution


clear
close all


%% Parameters

input_dim=20;
num_data=10000;

output_dim=input_dim;

net_struct=10*ones(1,10);
net_struct= [net_struct, output_dim];
neurons=cell(1,length(net_struct));


% neuron_func=@(z)tanh(z);
neuron_func=@(z)sigmf(z, [1,0]);


%% Stochastic parameters

mu=1;
sigma=1/3;  % to make sure it is between -1 and 1 with high probability

% data generation

%independent
% X=sigma*randn(num_data, input_dim)+mu;
% X=sigma*rand(num_data, input_dim)+mu;


% correlated
% R=0.2*sigma*ones(input_dim);
% % R=0.2*sigma*(2*rand(input_dim)-1);
% for i=1:input_dim
%     R(i,i)=sigma^2;
% end
% L = chol(R);
% X = X*L;


%Totally dependent

X=zeros(num_data, input_dim);
X(:,1)=sigma*randn(num_data, 1)+mu;
for i=2:input_dim
   rand_power=randi(10);
   X(:,i)= X(:,1).^rand_power;      
end


% paremeters statistics
mu_w=0;
sigma_w=0.1;

mu_b=0;
sigma_b=0.01;


%% Net simulation

W=sigma_w*randn(input_dim, net_struct(1));
b=sigma_b*randn(1, net_struct(1));

neurons{1}=neuron_func( X*W+b);


for i=2:length(net_struct)
    
    W=sigma_w*randn(net_struct(i-1), net_struct(i));
    b=sigma_b*randn(1, net_struct(i))+mu_b;
    
    neurons{i}=neuron_func( neurons{i-1}*W+b);
    
      
end



%% ploting

% figure


for i=1:length(net_struct)
    for j=1:net_struct(i)
        
%         subplot( length(net_struct), max(net_struct), i+j-1)
        figure
        histogram ( neurons{i}(:,j)) 
    end
end






