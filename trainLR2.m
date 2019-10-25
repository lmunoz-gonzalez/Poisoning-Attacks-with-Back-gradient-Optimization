function [w] = trainLR2(x,y,rate,iter)
%Train a Logistic Regression classifier with momentum

n = size(x,1);
D = size(x,2);

w = zeros(D,1);

for i=1:iter
    s = sigmoid(x*w);
    dw = mean(repmat(s-y,1,D).*x)';
    w = w - rate*dw;    
end
