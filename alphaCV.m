function [alpha_hat] = alphaCV(mmv, A, prior, sig, L, fold, grid_size, low_bnd, up_bnd)
 
[dim, vec_num] = size(mmv); % dimension of domain, number of mmv

alpha_vec = linspace(low_bnd, up_bnd, grid_size); % alpha's from each test set
mse_vec = zeros(1, grid_size);

for ll = 1:grid_size % iterates over each gridpoint

    fold_size = idivide(int16(vec_num), int16(fold));
    fold_size = double(fold_size);
    mse_fold_vec = zeros(1, fold);
    
    for ii = 0:(fold-1) % iterate over each fold

        curr_fold_size = fold_size;
        if rem(vec_num, fold) >= ii
            curr_fold_size = curr_fold_size + 1;
        end
    
        training_x = zeros(dim, vec_num - curr_fold_size);
        testing_y = zeros(dim, curr_fold_size);
        training_counter = 1;
        testing_counter = 1;
        
        for jj=1:vec_num % MAP estimation of each training vector
                            % adds testing vectors
            if mod(jj, fold) ~= ii
                
                    cvx_begin quiet
                    clear cvx
                    variable x_map(dim,1)
                    minimize((1/(2*sig^2)) * norm(mmv(:, jj)-A*x_map,2) + alpha_vec(ll) * norm(L*x_map,prior));
                    cvx_end
   
                    training_x(:,training_counter) = x_map;
                    training_counter = training_counter +1;
            else
                    testing_y(:, testing_counter) = mmv(:, jj);
                    testing_counter = testing_counter + 1;
            end         %end of adding the vector to training or testing
        end % end of iteration for training vectors and testing vectors
        
%        training_x_mean = mean(training_x, 2);
       
%        mse_fold_mean = 0;
%        
%        for kk=1:(curr_fold_size)
%            mse_fold_mean = mse_fold_mean + (1./(vec_num - curr_fold_size))*(immse(A*training_x_mean, testing_y(:, kk)));
%        end
       
        % mean square error between recovered x from training data set
        % and the mean of the test fold data set
%        mse_fold_mean = immse(A*training_x_mean, mean(testing_y, 2));

        mse_fold_mean = 0;
        for hh=1:(vec_num - curr_fold_size)
            mse_fold_mean = mse_fold_mean + (1./(vec_num - curr_fold_size))*(immse(A*training_x(:, hh), mean(testing_y, 2)));
        end
        
        % adds mse of ii^th fold to the mse vector
       mse_fold_vec(ii+1) = mse_fold_mean;
       
    end % end of fold iteration
    
    mse_vec(ll) = mean(mse_fold_vec);
       
end % end of gridpoint iteration

figure;
plot(log10(alpha_vec), log10(mse_vec));

ind_min = mse_vec == min(mse_vec(:)); 
alpha_hat = mean(alpha_vec(ind_min)); % calculates the mean of alphas
                                        % corresponding to min mse