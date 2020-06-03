function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = size(X)(1,1);
n = size(X)(1,2);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

Sum = 0;
Sum_reg = 0;
for i=1:m
    Sum = Sum - y(i,1)*log(sigmoid(X(i,:)*theta)) - (1-y(i,1))*log(1-sigmoid(X(i,:)*theta));
end;
Sum = Sum/m;

for j=2:n
    Sum_reg = Sum_reg + theta(j,1)^2;    
end;
Sum_reg = (lambda/(2*m))*Sum_reg;


for j=1:n
    for i=1:m
        grad(j,1) = grad(j,1) + (sigmoid(X(i,:)*theta)-y(i,1))*X(i,j);
    end; 

    if (j>1)
        grad(j,1) = grad(j,1) + (lambda*theta(j,1));
    endif;
end;

J = Sum + Sum_reg;
grad = (1/m)*grad;
% =============================================================
% grad = grad(:);
end
