% MAP estimation
% Theresa Scarnati
% modified by Jiahui (Jack) Zhang (July 2020)

clear all
close all

%% parameters

prior =1; %p of the ell_p prior
N = 80; % grid size
J = 20; % number of mmvs
PA_order = 2; % PA order
sig = 0.25; % standard deviation of AGWN
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

snr_y = zeros(J, 1);

mmv = zeros(N, J);

% building the J measurement vectors
for jj=1:J
    eta = sig*(randn(N,1));%+1i*randn(N,1)); % AGWN
    y = A*x + eta; % data
    mmv(:,jj) = y;
    snr_y(jj,1) = snr(A*x, eta);
end

mmv_mean = mean(mmv, 2);
fprintf('The unweighted signal-to-noise ratio is: %2.2f \n', mean(snr_y));
%%
% J MAP estimates

fprintf('Calclulating MAP estimates\n');

L = PA_Operator_1D(N,PA_order); %polynomial annihiliation operator
x_tilde = zeros(N,J);
PAx_map = zeros(N,J);

alpha_vec = linspace(0, 2, 10); % there needs to be two for-loops (variance and alpha)

reg_param = 2 * var_est; % alpha * sigma^2
% initial MAP reconstructions
tic;
for jj = 1:J
    cvx_begin quiet
    clear cvx
    variable x_map(N,1)
    minimize((1/reg_param) * norm(mmv(:, jj)-A*x_map,2)+ norm(L*x_map,prior));
    cvx_end
    
    x_tilde(:,jj) = x_map; % signal reconstructions
    PAx_map(:,jj) = L*x_map; % function in sparsity domain
    
end
time(1) = toc;
error_map = norm(x-mean(x_tilde,2))./norm(x);

% error_vec = zeros(J, 1);
% error_inv_sum = sum(error_vec.^(-1));