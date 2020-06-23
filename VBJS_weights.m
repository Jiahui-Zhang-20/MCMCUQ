function [W, var_vec] = VBJS_weights(f_tilde)
% VBJS weights
%
% Inputs:
% f_tilde = the mmv in sparse domain (N x J matrix)
%
% Output:
% W = diagonal matrix of weights
% var_vec = variances (J dimensional vector
%
%


[N,J] = size(f_tilde);  %dimensions of MMV
L = PA_Operator_1D(N, 2);   %polynomial annihilator
P = zeros(N, J);    % initialization of normalized MMV in edge domain
tau = 0.2;  %threshold

% transforms  f_tilde to sparse (edge) domain
for ii=1:J
    P(:, ii) = L*f_tilde(:, ii);
end

% normalizes each column  vectori n P
for ii=1:J
    P(:, ii) = abs(P(:, ii))./max(abs(P(:, ii)));
end

% initialize variance vector
var_vec = zeros(N, 1);

% tabulate variance vector
for ii=1:N
    var_vec(ii, 1) = var(P(ii, :));
end

% normalizes variance vector
var_vec_normed = var_vec ./ norm(var_vec, inf);

% calculates  normalization coefficient
C= 1/J * sum(reshape(P',[],1));

% initializes the indicator vector
I= zeros(N,1);

% all locations assumed to have edge (based on tau)
% are have 1 in the indicator vector
for ii=1:N
    if (1/J)*sum(P(ii, :)) > tau
        I(ii) = 1;
    end
end

%initialize the weights vector
weights = zeros(1, N);

% build weights
for ii=1:N
    if I(ii) == 0
        weights(1, ii) = C*(1-var_vec_normed(ii, 1));
    else
        weights(1, ii) = (1./C)*(1-var_vec_normed(ii, 1));
    end
end

% form diagonal N x N matrix with weights
W = diag(weights);

end

