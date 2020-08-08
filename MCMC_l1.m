% MCMC for Uncertainty  Quantification
% Theresa Scarnati
% modified by Jiahui (Jack) Zhang (July 2020)

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
sig = 0.25; % standard deviation of AGWN
prop_var = 0.1; % proposal distribution variance (Gaussian)
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

%% noise variance estimation

var_est = mean(var(mmv, 0, 2)); % variances along the rows (each vector is a sample of Y)
fprintf('the estimated standard deviation of the noise is: %f \n ', sqrt(var_est));

%%
% J MAP estimates

fprintf('Calclulating MAP estimates\n');

L = PA_Operator_1D(N,PA_order); %polynomial annihiliation operator
x_tilde = zeros(N,J);
PAx_map = zeros(N,J);

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

error_vec = zeros(J, 1);
error_inv_sum = sum(error_vec.^(-1));

%% calculate weights

[W, var_vec] = VBJS_weights(x_tilde);
%% unweighted posterior
var_pos = 2* var_est; % alpha  * sigma^2
f_post = @(x) exp(-(1/var_pos)* norm(mmv(:, 1)-A*x,2)^2 - norm(L*x,prior)); % f_post:R^N->R
%% unweighted MCMC
fprintf('Building MCMC chains...\n');

x_MH = zeros(N,N_M); % each row is a MC for the jth grid point
prop = @(x) normrnd(x, prop_var); % Gaussian proposal distribution - symmetric

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
%% weighted posterior
var_pos_w = 2* var_est; % alpha * sigma^2
f_post_w = @(x) exp(- (1/var_pos) * norm(mmv(:, 1)-A*x,2)^2 - norm(W*L*x,prior)); % f_post_w:R^N->R

%% Weighted MCMC

x_MH_w = zeros(N,N_M); % each row is a MC for the jth grid point
% prop = @(x) (x-beta/2) + (beta)*rand(N,1); % uniform proposal distribution - symmetric
prop = @(x) normrnd(x, prop_var); % Gaussian proposal distribution - symmetric

% vector of the diagonal entries
weights = diag(W);

f3=figure;
plot(grid, weights,'ob','linewidth',1.5);
title("Weights");

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
title("Autocorrelation at h(x) = 40");
subplot(2, 2, 2);
x_acf_0 = acf(x_mean_0', lag_num);
hold on; xline(BI);
title("Autocorrelation at h(x) = 0");
subplot(2, 2, 3);
x_acf_10 = acf(x_mean_10', lag_num);
hold on; xline(BI);
title("Autocorrelation at h(x) = 10");
subplot(2, 2, 4);
x_acf_50 = acf(x_mean_50', lag_num);
hold on; xline(BI);
title("Autocorrelation at h(x) = 50");
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
fprintf('Calculating credibility intervals...\n');

tic
% unweighted MCMC CI
ci = zeros(2,N);
for ii = 1:N
    
    dat = x_MH(ii,BI:end);
    
    SEM = std(dat)/sqrt(length(dat));               % Standard Error

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
sgtitle('Trace Plots for Unweighted MCMC'); 

f12=figure; 
plot(accept_ratio(1:end))
xline(BI);
title('Acceptance Ratio for Unweighted MCMC')

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
sgtitle('Trace Plots for Weighted MCMC'); 

f14=figure;  
plot(accept_ratio_w(1:end))
xline(BI);
title('Acceptance Ratio for Weighted MCMC')


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

%% write the specifications of the MCMC run
specfilename = sprintf('./Figures/%s/Specs.txt', currDate);
specfile = fopen(specfilename, 'wt' );
fprintf(specfile, 'MCMC Specifications: \n');
fprintf(specfile, '\n');
fprintf(specfile, 'Grid size: %d\n', N);
fprintf(specfile, 'Number of multiple measurement vectors: %d\n', J);
fprintf(specfile, 'Prior distribution: ell_%d\n', prior);
fprintf(specfile, 'Coefficient on the prior: %.3f\n', var_pos);
fprintf(specfile, 'Polynomial annihilation order: %d\n', PA_order);
fprintf(specfile, 'MCMC chain length: %d\n', N_M);
fprintf(specfile, 'Burn-in length: %d\n', BI);
fprintf(specfile, 'Signal to noise ratio: %.4f\n', mean(snr_y));
fprintf(specfile, 'Proposal distribution variance: %.3f\n', prop_var);
fprintf(specfile, 'Unweighted MCMC acceptance rate: %d/%d \n', num_accept, N_M);
fprintf(specfile, 'Weighted MCMC acceptance rate: %d/%d \n', num_accept_w, N_M);
fclose(specfile);
