function [cost,dw] = getDerivativesMLP(x,y,net)

n = size(x,1);
D = size(x,2) + 1;

x = [x ones(n,1)];
%Forward propagation
if (net.act == 1)
    s1 = sigmoid(x*net.w);
end
if (net.act == 2)
    s1 = tanh(x*net.w);
end
s1 = [s1 ones(n,1)];
s = sigmoid(s1*net.w_2);
%Cost
cost = mean(LRcost(s,y));
%Derivatives
delta_output = (s - y);
if (net.act == 1)
    delta_input = net.w_2*delta_output'.*sigmoidGradient([x*net.w ones(size(x,1),1)]'); 
else
    delta_input = net.w_2*delta_output'.*tanhGradient([x*net.w ones(size(x,1),1)]');
end
delta_input = delta_input(1:end-1,:);
grad2 = s1'*delta_output./n;
grad = x'*delta_input'./n;

grad = grad(:);

dw = [grad; grad2];


end

%Gradient of the sigmoid function
function [g] = sigmoidGradient(z)
%g = zeros(size(z));
g = sigmoid(z).*(1-sigmoid(z));
end

%Gradient of tanh function
function [g] = tanhGradient(z)
g = 1 - (tanh(z)).^2;
end



