%% Create training/validation/test splits with MNIST dataset

load 'mnist_1_7'

n = length(y);

for i=1:10
    perm = randperm(n);
    x = x(perm,:);
    y = y(perm);
    x_tr = x(1:100,:);
    y_tr = y(1:100,:);
    x_val = x(101:500,:);
    y_val = y(101:500);
    x_tst = x(501:end,:);
    y_tst = y(501:end);
    name = strcat('mnist_',num2str(i));
    save(name,'x_tr','y_tr','x_val','y_val','x_tst','y_tst');
end