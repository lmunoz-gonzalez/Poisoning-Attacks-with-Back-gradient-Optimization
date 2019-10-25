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
    %Change labels to -1/1
    y_tr = y_tr.*2 - 1;
    y_val = y_val.*2 - 1;
    y_tst = y_tst.*2 - 1;
    
    x_tr_e = [ones(size(x_tr,1),1) x_tr];
    x_val_e = [ones(size(x_val,1),1) x_val];
    x_tst_e = [ones(size(x_tst,1),1) x_tst];
    %Train Adaline and compute validation and test errors
    [w] = trainAdaline(x_tr_e,y_tr,0.01,2000);
    s = (x_val_e*w);
    t = sign(s);
    error_clean_val = mean(t~=y_val);
    s = (x_tst_e*w);
    t = sign(s);
    error_clean_test = mean(t~=y_tst);
    error_val(r,1) = error_clean_val;
    error_test(r,1) = error_clean_test;
    pfa_test(r,1) = sum(t == 1 & y_tst == -1)./sum(y_tst==-1);
    pm_test(r,1) = sum(t == -1 & y_tst == 1)./sum(y_tst==1);
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
            yp = -1;
        else
            yp = 1;
        end
        xp0 = xp;

        iter = 100;
        mu = 0.02;
        cost = zeros(iter,1);
        alpha = 1;
        
        cost_old = 0;
        flag = 0;
        i = 1;
        while(i < iter & flag == 0) 
            [cost(i),dxp] = reverseAdaline(xp',yp,x_tr,y_tr,x_val,y_val,60,mu);
            dxp = dxp./norm(dxp);
            xp = xp + alpha.*dxp';
            xp(xp > 1) = 1;
            xp(xp < 0) = 0;
            cost_iteration = cost(i);
            if (i == 1)
                fprintf('Iter %d : cost %1.4f\n',i,cost(i));
            end
            if (mod(i,10)==0)
                fprintf('Iter %d : cost %1.4f\n',i,cost(i));
            end
            if (cost_iteration - cost_old < 1e-5 & i > 30)
                fprintf('Finished at iter %d : cost %1.4f\n',i,cost(i));
                flag = 1;
            end
            cost_old = cost_iteration;
            i = i+1;
        end
        
        x_tr = [x_tr; xp];
        y_tr = [y_tr; yp];
        x_tr_e = [ones(size(x_tr,1),1) x_tr];
        [w] = trainAdaline(x_tr_e,y_tr,0.01,2000);

        s = (x_val_e*w);
        t = sign(s);
        error_it_val = mean(t~=y_val);
        s = (x_tst_e*w);
        t = sign(s);
        error_it_test = mean(t~=y_tst);
        error_val(r,j+1) = error_it_val;
        error_test(r,j+1) = error_it_test;
        pfa_test(r,j+1) = sum(t == 1 & y_tst == -1)./sum(y_tst==-1);
        pm_test(r,j+1) = sum(t == -1 & y_tst == 1)./sum(y_tst==1);
        fprintf('Error val: %1.4f\n',error_val(r,j+1));
        fprintf('Error test: %1.4f\n',error_test(r,j+1));
        fprintf('PFA test: %1.4f\n',pfa_test(r,j+1));
        fprintf('PM test: %1.4f\n\n',pm_test(r,j+1));
        
        name = strcat('MNIST_splits/ResultsAdalineMNIST');
        save(name,'error_test','error_val','pfa_test','pm_test');
    end
end



