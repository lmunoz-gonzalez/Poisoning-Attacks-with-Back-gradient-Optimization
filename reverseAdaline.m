function [cost,dxp] = reverseAdaline(xp,yp,x,y,xval,yval,iter,alpha);

xp = xp';
xp_e = [1 xp];

D = size(x,2) + 1;
n = size(x,1) + 1;
nval = size(xval,1);

x2 = [x; xp];
x2_e = [ones(size(x2,1),1) x2];
xval_e = [ones(nval,1) xval];
y2 = [y; yp];

%Train the classifier up to a certain number of iterations
% iter = 80;
% alpha = 0.05;
[w] = trainAdaline(x2_e,y2,alpha,iter);
w_T = w;

%Initialize reverseLR
dxp = zeros(size(xp));
dw = mean(repmat((xval_e*w - yval),1,size(xval_e,2)).*xval_e);

epsilon = 1e-8;

for i=1:(iter-1)
      
    %Gradient ascent
    g = mean(repmat(x2_e*w - y2,1,D).*x2_e)';
    w = w + alpha.*g;
    
    %Approximation (Pearlmutter)
    w2 = w + epsilon.*dw';   
    ddw2 = (repmat((x2_e(end,:)*w2 - y2(end)),1,size(x2,2)).*w2(2:end)')./n;
    ddw1 = (repmat((x2_e(end,:)*w - y2(end)),1,size(x2,2)).*w(2:end)')./n;
    app2 = (ddw2 - ddw1)./epsilon;
    dxp = dxp - alpha.*app2;
        
    %Approximation (Pearlmutter)
    w2 = w + epsilon.*dw';   
    ddw2 = mean(repmat((x2_e*w2 - y2),1,size(x2_e,2)).*x2_e);
    ddw1 = mean(repmat((x2_e*w - y2),1,size(x2_e,2)).*x2_e);
    app = (ddw2 - ddw1)./epsilon;
    dw = dw - alpha.*app;        

end


dxp = dxp';
cost = mean((xval_e*w_T - yval).^2);
