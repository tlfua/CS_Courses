function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
% m = length(y); % number of training examples
m = size(X)(1,1);
n = size(X)(1,2);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% g = zeros(size(theta));
% g_reg = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% ====    J
Sum = (X*theta - y)'*(X*theta - y);
Sum = Sum/(2*m);

Sum_reg = theta(2:end)'*theta(2:end);
Sum_reg = lambda*Sum_reg/(2*m);

J = Sum + Sum_reg; 

% ====    grad
for j=1:n
    for i=1:m
        grad(j,1) = grad(j,1) + (X(i,:)*theta - y(i,1))*X(i,j); 
    end;

    if j!=1
        grad(j,1) = grad(j,1) + lambda*theta(j,1);
    endif;
end;
grad = (1/m)*grad;
grad = grad(:);

end
