% MAP estimation (error vs. alpha plotting)
% modified by Jiahui (Jack) Zhang (August 2020)

clear all
close all

%% parameters

prior =1; %p of the ell_p prior
N = 80; % grid size
J = 10; % number of mmvs
PA_order = 2; % PA order
sig = 0.25; % standard deviation of AGWN

M = 10;  % number of different noise levels
noise_arr = linspace(0.2, 2.0, M);
K = 20; % number of tuning parameters
alpha_vec = linspace(0.05, 1.0, K); % vector of tuning parameters
%% function to be approximated
a = 0;
b = 1;
dx = (b-a)./(N-1);
grid = a + dx*(0:N-1)';

x = zeros(N,1);
x = x + 40.*(grid>0.1&grid<0.25) + 10.*(grid<0.35&grid>0.325) + ...
    (2*pi./(sqrt(2*pi)*0.05)*exp(-((grid-0.75)/0.05).^2/(2))).*(grid>0.5);
%% forward model

% blurring
h = 1/N;
gamma = .05;
A = zeros(N,N);
for ii = 1:N
    for jj =1:N
        A(ii,jj) = h.*exp(-((ii-jj)*h)^2/(2*gamma^2))/sqrt(pi*gamma^2);
    end
end

% random forward model
A = 0.1*randn(N,N);
%% data

% mmv = zeros(N, J);
mmv_arr = zeros(N, M, J); % M noise levels with J N-dim arrays each

% building the J measurement vectors
for mm= 1:M % varying noise levels
%     snr_y = zeros(J, 1);
    for jj=1:J % iterate over all measurement vectors
        eta = sig*(randn(N,1)); 
        y = A*x + eta; % data
%     mmv(:,jj) = y;
        mmv_arr(:, mm, jj) = y;
%         snr_y(jj,1) = snr(A*x, eta);
    end
end

% mmv_mean = mean(mmv, 2);
% fprintf('The unweighted signal-to-noise ratio is: %2.2f \n', mean(snr_y));

%% J MAP estimates

fprintf('Calclulating MAP estimates\n');

L = PA_Operator_1D(N,PA_order); %polynomial annihiliation operator

error_arr = zeros(K, M); % M columns: each column a plot of err vs. alpha

reg_param = 2 * sig^2; % alpha * sigma^2

% initial MAP reconstructions
tic;
for mm = 1:M % iterate through noise level
    
    for aa = 1:K % iterate through alpha's
        
             x_tilde = zeros(N,J);
%             PAx_map = zeros(N,J);
        for jj = 1:J % iterate over each measurement vector
            
            cvx_begin quiet
            clear cvx
            variable x_map(N,1)
            minimize((1/reg_param) * norm(mmv_arr(:, mm, jj)-A*x_map,2)+ alpha_vec(aa) *norm(L*x_map,prior));
            cvx_end
    
            x_tilde(:,jj) = x_map; % signal reconstructions
%           PAx_map(:,jj) = L*x_map; % function in sparsity domain
        end
        
        error_map = norm(x-mean(x_tilde, 2)); %l_2 distance between two vecs
        error_arr(aa, mm)  = error_map;
    end
    
end


% plotting error vs. alpha for each noise level
fig = figure();
Legend = cell(M, 1);
hold on;
for mm = 1:M
    plot(alpha_vec, error_arr(:, mm)); hold on
    Legend{mm}=strcat('Noise SD: ', num2str(noise_arr(mm)));
end
legend(Legend);
title = ('l_2 distance vs. alpha');
xlabel('alpha');
ylabel('l2 error');