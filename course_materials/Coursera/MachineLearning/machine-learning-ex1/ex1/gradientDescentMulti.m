function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
S = size(X);
n = S(1,2);


fprintf("m=%f, n=%f\n",m,n);

J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    Sum = zeros(n, 1);

    for j=1:n
        for i=1:m
            Sum(j,1) = Sum(j,1) + ((theta'*X(i,:)')-y(i))*X(i,j);
        end;
    end;

    Sum = (alpha/m)*Sum;
    theta = theta - Sum;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
