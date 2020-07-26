% MCMC for Uncertainty  Quantification
% Theresa Scarnati
% modified by Jiahui (Jack) Zhang (June 2020)

clear all
close all

% creates subdirectory "Figures" if it does not already exist
if ~exist('./Figures', 'dir')
	mkdir("Figures")
end

% adds a folder within Figures for figures generated from this execution
addpath(genpath('../helper_functs'));
currDate = strrep(datestr(datetime), ':', '_');
mkdir('./Figures',currDate)

%% parameters

prior =1; %p of the ell_p prior
N = 80; % grid size
J = 20; % number of mmvs
PA_order = 2; % PA order
N_M = 50000; % Chain length
BI = 25000; % burn in length
sig =1.00; % standard deviation of AGWN
prop_var = 0.1;

beta = 1.0; % width of proposal distribution

%% function to be approximated
a = 0;
b = 1;
dx = (b-a)./(N-1);
grid = a + dx*(0:N-1)';

x = zeros(N,1);
x = x + 40.*(grid>0.1&grid<0.25) + 10.*(grid<0.35&grid>0.325) + ...
    (2*pi./(sqrt(2*pi)*0.05)*exp(-((grid-0.75)/0.05).^2/(2))).*(grid>0.5);

% plots the true function and it's edge transform
f1=figure;
plot(grid,x,'--k','linewidth',1.5);
title('True Function');

f2=figure;
L = PA_Operator_1D(N,PA_order); %polynomial annihiliation operator
plot(grid, L*x, '-k', 'linewidth', 1.5);
title('Edge Domain')


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

%  A = dftmtx(N)./sqrt(N); 

%% data

% eta = sig*(randn(N,1));%+1i*randn(N,1)); % AGWN
% y = A*x + eta; % data

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

% figure;
% plot(grid,x,'--k','linewidth',1.5);
% ylim([-10,50]);
% L = legend('True');
% set(L,'interpreter','latex','fontsize',14,'location','north')
% title('True function')

% figure;
% plot(grid,x,'--k','linewidth',1.5);
% hold on;
% plot(grid,mmv_mean,'ob','linewidth',1.5);
% ylim([-10,50]);
% L = legend('True','Data');
% set(L,'interpreter','latex','fontsize',14,'location','north')
% title('True function and Data with Noise')

% %% MMV
% 
% y_mmv = zeros(N, J);
% 
% for ii = 1:J
%     % random forward model
%     A = 0.1*randn(N,N);
%     eta = sig*(randn(N,1));%+1i*randn(N,1)); % AGWN
%     y_mmv(:, ii) = A*x + eta; % data
% end

%%
% J MAP estimates

fprintf('Calclulating MAP estimates\n');

L = PA_Operator_1D(N,PA_order); %polynomial annihiliation operator
%lam = zeros(J,1);
x_tilde = zeros(N,J);
PAx_map = zeros(N,J);

% figure;
% plot(grid,x,'--k'); hold on;
% title('True Function')

% initial MAP reconstructions
tic;
for jj = 1:J
    %lam(jj) = rand;
    %lam(jj) = (ii)/(J);
    cvx_begin quiet
    clear cvx
    variable x_map(N,1)
    minimize( norm(mmv(:, jj)-A*x_map,2)+0.5*norm(L*x_map,prior));
    %minimize( norm(y_mmv(:, ii)-A*x_map,2)+0.5*norm(W*L*x_map,1));

    cvx_end
    
%     plot(grid,x_map);
    
    x_tilde(:,jj) = x_map; % signal reconstructions
    PAx_map(:,jj) = L*x_map; % function in sparsity domain
    
end
time(1) = toc;
error_map = norm(x-mean(x_tilde,2))./norm(x);

error_vec = zeros(J, 1);
error_inv_sum = sum(error_vec.^(-1));

%% calculate weights

[W, var_vec] = VBJS_weights(x_tilde);

    
% for kk = 1:J
%     error = norm(x-x_tilde(:, kk))./norm(x);
%     error_vec(kk) = error;
%     fprintf('\n The error of MAP estimates with lambda = %.2d: %g \n', lam(kk), error_vec(kk));
% 
% end

%lambda_weighted = sum(lam./error_vec)./(sum(1./error_vec));
%fprintf("The weighted lambda: %1.2f \n", lambda_weighted);
%fprintf('MAP \t || time = %2.2f sec \t|| error = %2.4f \n',time(1),error_map);


%% unweighted posterior

%var = mean(lam); % just my first idea (we could change this)
var_pos = .25;
%f_post = @(x) exp(-norm(y-A*x,2)^2 - lambda_weighted*norm(L*x,1)); % f_post:R^N->R
f_post = @(x) exp(-norm(mmv(:, 1)-A*x,2)^2 - var_pos*norm(L*x,prior)); % f_post:R^N->R


%% unweighted MCMC
fprintf('Building MCMC chains...\n');

x_MH = zeros(N,N_M); % each row is a MC for the jth grid point
% prop = @(x) (x-beta/2) + (beta)*rand(N,1); % uniform proposal distribution - symmetric
prop = @(x) normrnd(x, prop_var); % Gaussian proposal distribution - symmetric

% prop = @(x) x + beta*randn; 

x_MH(:,1) = mean(x_tilde,2); % initial condition

num_reject = 0; 
num_accept = 0; 
accept_ratio = zeros(N_M,1); 
log_post = zeros(N_M,1); 

% metropolis hastings
tic
for k = 2:N_M
    
    x_cand = prop(x_MH(:,k-1));
    ratio = (f_post(x_cand)./f_post(x_MH(:,k-1)));
    alpha = min(1,ratio);
    
    u = rand;
    
    if u<alpha
        x_MH(:,k) = x_cand;
        num_accept = num_accept + 1; 

    else
        x_MH(:,k) = x_MH(:,k-1);
        num_reject = num_reject + 1; 
    end
    
    log_post(k) = log(f_post(x_MH(:,k))); 
    accept_ratio(k) = num_accept/k;
    
end

time(2) = toc;

error_mcmc = norm(x-mean(x_MH(:,BI:end),2))./norm(x);

fprintf('Unweighted MCMC \t || time = %2.2f sec \t|| error = %2.4f \t|| accept = %d/%d \n',time(2),error_mcmc,num_accept,N_M);


% %% sparse domain of MCMC (mean and variance)
% L = PA_Operator_1D(N,PA_order); %polynomial annihiliation operator
% x_MH_sparse = L*(x_MH(:, BI:end));
% x_MH_s_m = mean(x_MH_sparse, 2);
% x_MH_s_v = var(x_MH_sparse');

% %% calculate weights
% 
% weights = zeros(1, N);
% for ii=1:N
%     weights(ii) = 1/x_MH_s_v(ii);
% end
% 
% weights_normed = norm(weights, 1);
% W = diag(weights);

% %% weighted MAP estimates
% 
% L = PA_Operator_1D(N,PA_order); %polynomial annihiliation operator
% lam = zeros(J,1);
% x_tilde_w = zeros(N,J);
% PAx_map_w = zeros(N,J);
% 
% % figure;
% % plot(grid,x,'--k'); hold on;
% % title('True Function')
% 
% % initial MAP reconstructions
% tic;
% for ii = 1:J
%     %lam(ii) = rand;
%     lam(ii) = (ii)/(J);
%     cvx_begin quiet
%     clear cvx
%     variable x_map_w(N,1)
%     minimize( norm(y-A*x_map_w,2)+lam(ii)*norm(W*L*x_map_w,1));
%     cvx_end
%     
% %     plot(grid,x_map);
%     
%     x_tilde_w(:,ii) = x_map_w; % signal reconstructions
%     PAx_map_w(:,ii) = L*x_map_w; % function in sparsity domain
%     
% end

%% weighted posterior

%var = mean(lam); % just my first idea (we could change this)
var_pos_w = 0.25;
%f_post = @(x) exp(-norm(y-A*x,2)^2 - lambda_weighted*norm(L*x,1)); % f_post:R^N->R
f_post_w = @(x) exp(-norm(mmv(:, 1)-A*x,2)^2 - var_pos_w*norm(W*L*x,prior)); % f_post:R^N->R

%% Weighted MCMC

x_MH_w = zeros(N,N_M); % each row is a MC for the jth grid point
% prop = @(x) (x-beta/2) + (beta)*rand(N,1); % uniform proposal distribution - symmetric
prop = @(x) normrnd(x, prop_var); % Gaussian proposal distribution - symmetric

% vector of the diagonal entries
weights = diag(W);

f3=figure;
plot(grid, weights,'ob','linewidth',1.5);
title("Weights");

% prop = @(x) x + beta*randn; 

x_MH_w(:,1) = mean(x_tilde,2); % initial condition

num_reject_w = 0; 
num_accept_w = 0; 
accept_ratio_w = zeros(N_M,1); 
log_post_w = zeros(N_M,1); 

% metropolis hastings
tic
for k = 2:N_M
    
    x_cand = prop(x_MH_w(:,k-1));
    ratio = (f_post_w(x_cand)./f_post_w(x_MH_w(:,k-1)));
    alpha = min(1,ratio);
    
    u = rand;
    
    if u<alpha
        x_MH_w(:,k) = x_cand;
        num_accept_w = num_accept_w + 1; 

    else
        x_MH_w(:,k) = x_MH_w(:,k-1);
        num_reject_w = num_reject_w + 1; 
    end
    
    log_post_w(k) = log(f_post_w(x_MH_w(:,k))); 
    accept_ratio_w(k) = num_accept_w/k;
    
end

time(3) = toc;

error_mcmc_w = norm(x-mean(x_MH_w(:,BI:end),2))./norm(x);

fprintf('Weighted MCMC \t || time = %2.2f sec \t|| error = %2.4f \t|| accept = %d/%d \n',time(3),error_mcmc_w,num_accept_w,N_M);

%% autocorrelation
% https://www.mathworks.com/matlabcentral/fileexchange/30540-autocorrelation-function-acf
fprintf('Calculating autocorrelation...\n');

lag_num = 5000;

% unweighted MCMC autocorrelation

ind_40 = find(x==40); 
ind_0 = find(x == 0); 
ind_10 = find(x==10); 
ind_50 = find(and(x<51,x>49));

x_mean_40 = mean(x_MH(ind_40,1:end));
x_mean_0 = mean(x_MH(ind_0,1:end));
x_mean_10 = mean(x_MH(ind_10,1:end));
x_mean_50 = mean(x_MH(ind_50,1:end));

f4=figure;
subplot(2, 2, 1);
x_acf_40 = acf(x_mean_40', lag_num);
hold on; xline(BI);
title("Autocorrelation at x = 40");
subplot(2, 2, 2);
x_acf_0 = acf(x_mean_0', lag_num);
hold on; xline(BI);
title("Autocorrelation at x = 0");
subplot(2, 2, 3);
x_acf_10 = acf(x_mean_10', lag_num);
hold on; xline(BI);
title("Autocorrelation at x = 10");
subplot(2, 2, 4);
x_acf_50 = acf(x_mean_50', lag_num);
hold on; xline(BI);
title("Autocorrelation at x = 50");
sgtitle("Autocorrelation of Unweighted MCMC Mean")

% weighted MCMC autocorrelation

x_mean_40_w = mean(x_MH_w(ind_40,BI:end));
x_mean_0_w = mean(x_MH_w(ind_0,BI:end));
x_mean_10_w = mean(x_MH_w(ind_10,BI:end));
x_mean_50_w = mean(x_MH_w(ind_50,BI:end));

f5=figure;
subplot(2, 2, 1);
x_acf_40_w = acf(x_mean_40_w', lag_num);
hold on; xline(BI);
title("Autocorrelation: h(x) = 40");
subplot(2, 2, 2);
x_acf_0_w = acf(x_mean_0_w', lag_num);
hold on; xline(BI);
title("Autocorrelation: h(x) = 0");
subplot(2, 2, 3);
x_acf_10_w = acf(x_mean_10_w', lag_num);
hold on; xline(BI);
title("Autocorrelation: h(x) = 10");
subplot(2, 2, 4);
x_acf_50_w = acf(x_mean_50_w', lag_num);
hold on; xline(BI);
title("Autocorrelation: h(x) = 50");
sgtitle('Autocorrelation of Weighted MCMC'); 


sgtitle("Autocorrelation of Weighted MCMC Mean")
%% credibility intervals
fprintf('Calculating confidence intervals...\n');

tic
% unweighted MCMC CI
ci = zeros(2,N);
for ii = 1:N
    
    dat = x_MH(ii,BI:end);
    
    SEM = std(dat)/sqrt(length(dat));               % Standard Error
%     ts = tinv([0.025  0.975],length(dat)-1)';     % T-Score
%     ci(:,ii) = mean(dat) + ts*SEM;                % Confidence Intervals 

    ci(1,ii) = quantile(dat,0.025);
    ci(2,ii) = quantile(dat,0.975);
end

% calculating weighted interval widths
x_ci_40 = mean(ci(2, ind_40) - ci(1, ind_40));
x_ci_0 = mean(ci(2, ind_0) - ci(1, ind_0));
x_ci_10 = mean(ci(2, ind_10) - ci(1, ind_10));
x_ci_50 = mean(ci(2, ind_50) - ci(1, ind_50));

time(3) = toc;

% weighted MCMC CI
ci_w = zeros(2, N);
for jj = 1:N
    
    dat_w = x_MH_w(jj, BI:end);
    
    SEM_w = std(dat_w)/sqrt(length(dat_w));
    
    ci_w(1, jj) = quantile(dat_w, 0.025);
    ci_w(2, jj) = quantile(dat_w, 0.975);
end

% calculating weighted interval widths
x_ci_w_40 = mean(ci_w(2, ind_40) - ci(1, ind_40));
x_ci_w_0 = mean(ci_w(2, ind_0) - ci(1, ind_0));
x_ci_w_10 = mean(ci_w(2, ind_10) - ci(1, ind_10));
x_ci_w_50 = mean(ci_w(2, ind_50) - ci(1, ind_50));

bar_ci = [ x_ci_40, x_ci_w_40
           x_ci_0, x_ci_w_0
           x_ci_10, x_ci_w_10
           x_ci_50, x_ci_w_50 ];
       
f6=figure; % plots the CI's in a double bar graph
title('Credibility Intervals');
bar(bar_ci);
legend('Unweighted MCMC', 'Weighted MCMC', 'location', 'northeastoutside');
set(gca,'xticklabel',{'h(x)=40', 'h(x)=0', 'h(x)=10', 'h(x)=50'});

fprintf('|| time = %2.2f sec \n',time(3));
%% plot results

f7=figure;
plot(grid,x,'--k','linewidth',1.5)
hold on;
plot(grid,x_MH(:,1),'b-.','linewidth',1.5)
plot(grid,mean(x_MH(:,BI:end),2),'r-','linewidth',1.5);
L = legend('True','MAP Mean','Unweighted MCMC Mean');
set(L,'interpreter','latex','fontsize',14,'location','north')
title('Results with Map Mean')

f8=figure;
plot(grid,x,'--k','linewidth',1.5)
hold on
plot(grid,mean(x_MH(:,BI:end),2),'r-','linewidth',1.5)
plot(grid,ci(1,:),'r-.')
plot(grid,ci(2,:),'r-.')
L = legend('True','Unweighted MCMC Mean','95\% CI');
set(L,'interpreter','latex','fontsize',14,'location','north')
title('Results with Credibility Intervals')
% 
% figure;
% plot(grid,mean(x_MH_sparse, 2),'b-','linewidth',1.5);
% hold on
% plot(grid, ci_sparse(1, :), 'r-');
% plot(grid, ci_sparse(2, :), 'b-');
% L = legend('True', 'MCMC Sparse mean', '95\% CI');
% set(L, 'interpreter', 'latex', 'fontsize', 14, 'location', 'north');
% title('Results in the sparse domain');
% 
% figure;
% plot(grid,var(x_MH_sparse'), 'b-','linewidth',1.5);
% title('variance in sparse domain')
% 
% figure;
% plot(grid, x, '--k', 'linewidth', 1.5)
% hold on
% plot(grid, mean(x_tilde, 2), 'b-', 'linewidth', 1.5);
% plot(grid, mean(x_tilde_w, 2), 'r-', 'linewidth', 1.5);
% L = legend('True', 'unweighted MAP', 'weighted MAP');
% set(L, 'interpreter', 'latex', 'fontsize', 14, 'location', 'north');
% 

f9=figure;
plot(grid,x,'--k','linewidth',1.5)
hold on
plot(grid,mean(x_MH_w, 2),'b-','linewidth',1.5);
plot(grid,mean(x_MH, 2),'r-','linewidth',1.5);
L = legend('True', 'weighted MCMC', 'Unweighted MCMC');
set(L, 'interpreter', 'latex', 'fontsize', 14, 'location', 'north');
title('Unweighted Versus Weighted MCMC');

f10=figure;
plot(grid,x,'--k','linewidth',1.5)
hold on
plot(grid,mean(x_MH_w(:,BI:end),2),'b-','linewidth',1.5)
plot(grid,ci_w(1,:),'b-.')
plot(grid,ci_w(2,:),'b-.')
L = legend('True Function','Weighted MCMC Mean','95\% CI');
set(L,'interpreter','latex','fontsize',14,'location','north')
title('Results with Credbility Intervals')

error_inv = norm(real(A\y)-x)/norm(x);
tic 
inv_x = real(A\y);
t = toc; 
fprintf('inv(A) \t || time = %2.4f sec \t|| error = %2.4f \t|| cond(A) = %2.4e \t|| rank(A) = %d  \n',...
    t, error_inv, cond(A),rank(A));

%% Convergence checks 
% https://jellis18.github.io/post/2018-01-02-mcmc-part1/

% ind_40 = find(x==40); 
% ind_0 = find(x == 0); 
% ind_10 = find(x==10); 
% ind_50 = find(and(x<51,x>49)); 

it = BI:N_M;

% unweighted MCMC convergence checks
f11=figure;
subplot(2,2,1); 
plot(x_MH(ind_40(ceil(length(ind_40)/2)),1:end)); 
ylim([35,45]);
xline(BI);
hold on; plot(40*ones(N_M,1),'r')
title("h(x) = 40");
subplot(2,2,2); 
plot(x_MH(ind_0(ceil(length(ind_0)/2)),1:end)); 
ylim([-5,5]);
xline(BI);
hold on; plot(0*ones(N_M,1),'r')
title("h(x) = 0");
subplot(2,2,3); 
plot(x_MH(ind_10(ceil(length(ind_10)/2)),1:end)); 
ylim([5,15]);
xline(BI);
hold on; plot(10*ones(N_M,1),'r')
title("h(x) = 10");
subplot(2,2,4); 
plot(x_MH(ind_50(ceil(length(ind_50)/2)),1:end));
ylim([45,55]);
xline(BI);
hold on; plot(50*ones(N_M,1),'r')
title("h(x) = 50");
sgtitle('Trace Plots at Locations of Vector x for Unweighted MCMC'); 

f12=figure; 
% subplot(1,2,1); 
plot(accept_ratio(1:end))
xline(BI);
title('Acceptance Ratio for Unweighted MCMC')
% subplot(1,2,2); 
% plot(log_post(1:end)) 
% xline(BI);
% title('log-posterior for Unweighted MCMC');

% weighted MCMC convergence checks
f13=figure;
subplot(2,2,1); 
plot(x_MH_w(ind_40(ceil(length(ind_40)/2)),1:end)); 
ylim([35,45]);
xline(BI);
hold on; plot(40*ones(N_M,1),'r')
title("h(x) = 40");
subplot(2,2,2); 
plot(x_MH_w(ind_0(ceil(length(ind_0)/2)),1:end)); 
ylim([-5,5]);
xline(BI);
hold on; plot(0*ones(N_M,1),'r')
title("h(x) = 0");
subplot(2,2,3); 
plot(x_MH_w(ind_10(ceil(length(ind_10)/2)),1:end)); 
ylim([5,15]);
xline(BI);
hold on; plot(10*ones(N_M,1),'r')
title("h(x) = 10");
subplot(2,2,4); 
plot(x_MH_w(ind_50(ceil(length(ind_50)/2)),1:end)); 
ylim([45,55]);
xline(BI);
hold on; plot(50*ones(N_M,1),'r')
title("h(x) = 50");
sgtitle('Trace Plots at Locations of Vector x for Weighted MCMC'); 

f14=figure; 
% subplot(1,2,1); 
plot(accept_ratio_w(1:end))
xline(BI);
title('Acceptance Ratio for Weighted MCMC')
% subplot(1,2,2); 
% plot(log_post_w(1:end)) 
% xline(BI);
% title('log-posterior for Weighted MCMC'); 

%% save the figures to the subdirectory "Figures"

prefix = sprintf('Noise_%.3f_', sig);
saveas(figure(f1),[pwd '/Figures/', currDate, '/', prefix, 'TrueFunction.jpg']);
saveas(figure(f2),[pwd '/Figures/', currDate, '/', prefix, 'EdgeDomain.jpg']);
saveas(figure(f3),[pwd '/Figures/', currDate, '/', prefix, 'VBJSWeights.jpg']);
saveas(figure(f4),[pwd '/Figures/', currDate, '/', prefix, 'Auto_uw.jpg']);
saveas(figure(f5),[pwd '/Figures/', currDate, '/', prefix, 'Auto_w.jpg']);
saveas(figure(f6),[pwd '/Figures/', currDate, '/', prefix, 'CI_Widths.jpg']);
saveas(figure(f7),[pwd '/Figures/', currDate, '/', prefix, 'MAP_Results.jpg']);
saveas(figure(f8),[pwd '/Figures/', currDate, '/', prefix, 'CI_uw.jpg']);
saveas(figure(f9),[pwd '/Figures/', currDate, '/', prefix, 'uw_vs_w.jpg']);
saveas(figure(f10),[pwd '/Figures/', currDate, '/', prefix, 'CI_w.jpg']);
saveas(figure(f11),[pwd '/Figures/', currDate, '/', prefix, 'Trace_uw.jpg']);
saveas(figure(f12),[pwd '/Figures/', currDate, '/', prefix, 'AR_uw.jpg']);
saveas(figure(f13),[pwd '/Figures/', currDate, '/', prefix, 'Trace_w.jpg']);
saveas(figure(f14),[pwd '/Figures/', currDate, '/', prefix, 'AR_w.jpg']);
close all

