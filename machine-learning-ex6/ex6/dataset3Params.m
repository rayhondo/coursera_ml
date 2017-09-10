function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

error_min = inf;
error_table = [];
optimal_C = 0;
optimal_sigma = 0;


for i = 1:length(C_range)
	test_C = C_range(i);

	for j = 1:length(sigma_range)
		test_sigma = sigma_range(j);

		model = svmTrain(X, y, test_C, @(x1, x2) gaussianKernel(x1, x2, test_sigma)); 
		predictions = svmPredict(model, Xval);
		prediction_error = mean(double(predictions ~= yval));

		error_table = [error_table; C_range(i) sigma_range(j) prediction_error];
		if(prediction_error <= error_min)
			optimal_C = test_C;
			optimal_sigma = test_sigma;
			error_min = prediction_error;
		end
	end
end

C = optimal_C;
sigma = optimal_sigma;

% =========================================================================

end
