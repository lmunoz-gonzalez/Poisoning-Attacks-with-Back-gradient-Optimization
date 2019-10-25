function [s,t] = testMLP(net,x);
%Gives the output of the MLP (whose parameters are in structure "net"):
% - s: soft-output
% - t: hard-output

n = size(x,1);
x = [x ones(n,1)];
if (net.act == 1)
    s1 = sigmoid(x*net.w);
end
if (net.act == 2)
    s1 = tanh(x*net.w);
end
s1 = [s1 ones(n,1)];
s = sigmoid(s1*net.w_2);
t = double(s > 0.5);