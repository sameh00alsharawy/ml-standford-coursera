function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = size(y,2); % number of training examples

% You need to return the following variables correctly 
J = 0;
n = length(theta);
grad = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = sigmoid(X*theta);

J = ((-y'*log(h)-(1-y)'*log(1-h))/m) + (lambda / (2*m )) * (theta(2:n)' * theta(2:n) );

grad(1) = ( X(:,1)'*(h-y))/m;
grad(2:n) = (( X(:,2:n)'*(h-y) )/m) + (lambda / m ) * theta(2:n);
grad = grad(:,1);



% =============================================================

end
