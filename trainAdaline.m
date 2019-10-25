function [w] = trainAdaline(x,y,rate,iter)
%Train a Logistic Regression classifier

n = size(x,1);
D = size(x,2);

w = zeros(D,1);

for i=1:iter
    s = (x*w);
    dw = mean(repmat(s-y,1,D).*x)';
    w = w - rate*dw;    
%     cost(i) = mean((x*w - y).^2);
end

% plot(cost)
% pause
