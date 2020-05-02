function [J, grad] = costFunction(theta, X, y)
  % Initialize some useful values
  m = length(y); % number of training examples

  % You need to return the following variables correctly 
  J = 0;
  grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
  Cost1 = -1 * (y .* log(sigmoid(X * theta)));
  Cost2 = (1 - y) .* log(1- sigmoid(X * theta));
  %Compute cost function
  J = sum(Cost1 - Cost2) / m;
  grad = (X' * (sigmoid(X * theta) - y)) * (1/m);    
% =============================================================

end
