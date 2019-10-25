%% Test attack on MLP
clear all
clc

reps = 10; %Number of repetitions
np = 20; %Number of poisoning points

error_val = zeros(reps,np+1);
error_test = zeros(reps,np+1);
pfa_test = zeros(reps,np+1);
pm_test = zeros(reps,np+1);

for r = 1:reps

    name = strcat('MNIST_splits/MNIST_',num2str(r));
    load(name)
    
    alpha = 0.1; %Learning rate
    iter = 400; %Number of epochs
    param.M = 10; %Number of neurons
    param.alpha = alpha;
    param.nit = iter;
    param.init = 1; %Initialize weights from a given values
    param.act = 2; %Use tanh as activation function in the hidden layer
    d = size(x_tr,2);
    M = param.M;
    %Initialization of the weights
    w0 = randn(d+1,M).*0.1;
    w0_2 = randn(M+1,1).*0.1; 
    param.w = w0;
    param.w_2 = w0_2;

    [net, errtr] = trainMLP(x_tr,y_tr,param); 
    [sval,tval] = testMLP(net,x_val);
    error_clean_val = mean(tval~=y_val);
    error_val(r,1) = error_clean_val;
    [stst,ttst] = testMLP(net,x_tst);
    error_clean_test = mean(ttst~=y_tst);
    error_test(r,1) = error_clean_test;
    pfa_test(r,1) = sum(ttst == 1 & y_tst == 0)./sum(y_tst==0);
    pm_test(r,1) = sum(ttst == 0 & y_tst == 1)./sum(y_tst);
    fprintf('----------------------------------\n');
    fprintf('Repetition %d\n',r);
    fprintf('----------------------------------\n');
    fprintf('CLEAN DATASET\n');
    fprintf('Error val: %1.4f\n',error_clean_val);
    fprintf('Error test: %1.4f\n',error_clean_test);
    fprintf('PFA test: %1.4f\n',pfa_test(r,1));
    fprintf('PM test: %1.4f\n\n',pm_test(r,1));
    
    for j=1:np
        fprintf('Poisoning point number: %d\n',j);
        %Choose poisoning point at random from the validation set and flip
        %the label
        nval = size(x_val,1);
        c = randi(nval,1);
        xp = x_val(c,:);
        if (y_val(c) == 1)
            yp = 0;
        else
            yp = 1;
        end
        xp0 = xp;

        iter = 200;
        cost = zeros(iter,1);
        alpha = 0.2;
        param.nit = 200;
        
        for i=1:iter  
            [cost(i) dxp] = reverseMLP2(xp',yp,x_tr,y_tr,x_val,y_val,param);
            dxp = dxp./norm(dxp);
            xp = xp + alpha.*dxp;
            xp(xp > 1) = 1;
            xp(xp < 0) = 0;
            cost_iteration = cost(i);
            if (i == 1)
                fprintf('Iter %d : cost %1.4f\n',i,cost(i));
            end
            if (mod(i,50)==0)
                fprintf('Iter %d : cost %1.4f\n',i,cost(i));
            end
        end
        x_tr = [x_tr; xp];
        y_tr = [y_tr; yp];
        param.nit = 400;
        [net, errtr] = trainMLP(x_tr,y_tr,param); 
        [sval,tval] = testMLP(net,x_val);
        error_it= mean(tval~=y_val);
        error_val(r,j+1) = error_it;
        [stest,ttest] = testMLP(net,x_tst);
        error_it_test = mean(ttest~=y_tst);
        error_test(r,j+1) = error_it_test;
        pfa_test(r,j+1) = sum(ttest == 1 & y_tst == 0)./sum(y_tst==0);
        pm_test(r,j+1) = sum(ttest == 0 & y_tst == 1)./sum(y_tst);
        fprintf('Error val: %1.4f\n',error_val(r,j+1));
        fprintf('Error test: %1.4f\n',error_test(r,j+1));
        fprintf('PFA test: %1.4f\n',pfa_test(r,j+1));
        fprintf('PM test: %1.4f\n\n',pm_test(r,j+1));
        
        name = strcat('MNIST_splits/ResultsMLPMNIST');
        save(name,'error_test','error_val','pfa_test','pm_test');
    end
end



