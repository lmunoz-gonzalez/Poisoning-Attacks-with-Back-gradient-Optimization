function [cost,dxp] = reverseMLP2(xp,yp,x,y,xval,yval,param);

xp = xp';

D = size(x,2) + 1;
n = size(x,1) + 1;
nval = size(xval,1);

x2 = [x; xp];
y2 = [y; yp];

iter = param.nit;
alpha = param.alpha;


[net, errtr,wv,gv] = trainMLP(x2,y2,param); 


dxp = zeros(size(xp));
[cost,dw] = getDerivativesMLP(xval,yval,net);

epsilon = 1e-8;

w = net.w;
w2 = net.w_2;

ww = [w(:); w2];
lw = length(w(:));


for i=1:(iter-1)    
    
    [c,dww] = getDerivativesMLP(x2,y2,net);
    ww = ww + alpha.*dww;
    wb = ww(1:lw);
    w2b = ww(lw+1:end);
    wb = reshape(wb,size(w));
    net.w = wb;
    net.w_2 = w2b; 
    
    
    wwm = ww;
    wwm = wwm + 0.5.*epsilon.*dw;
    wb = wwm(1:lw);
    w2b = wwm(lw+1:end);
    wb = reshape(wb,size(w));
    net2 = net;
    net2.w = wb;
    net2.w_2 = w2b;   
    [c2x,dw2x] = getDerivativesMLP2(x2,y2,net2);
    [c2,dw2] = getDerivativesMLP(x2,y2,net2);
     
    wwm = wwm - epsilon.*dw;
    wb = wwm(1:lw);
    w2b = wwm(lw+1:end);
    wb = reshape(wb,size(w));
    net1 = net;
    net1.w = wb;
    net1.w_2 = w2b;  
    [c1x,dw1x] = getDerivativesMLP2(x2,y2,net1);
    [c1,dw1] = getDerivativesMLP(x2,y2,net1);
    
    ddxp = (dw2x - dw1x)./epsilon;
    ddw = (dw2 - dw1)./epsilon;
    dxp = dxp - alpha.*ddxp';
    dw = dw - alpha.*ddw;
    
   
end


