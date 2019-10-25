function [net,errtr,wv,gv] = trainMLP(X_tr,Y_tr,param)

% INPUTS
%   - X_tr             Training samples [#samples #features]
%   - Param.
%       -M          number of neurons in the hidden layer
%       -alpha        learning rate    
%       -nit        number of iterations (epochs)
%       -act        activation function (1 - sigmoid, 2 - tanh)
%       -batch_size size of the mini-batch
%       -init       if 1, we store the initial weights for the NN in
%                   variables w, w_2
% OUTPUTS
%   - errtr         training error
%   - net.
%       -w          weights for the input layer
%       -w2         weights for the output layer
%       -fhidden    activation function in the hidden layer (1 - sigmoid,
%                    2 - tanh)
%//////////////////////////////////////////////////////////////////////////


%//////////////////////////////////////////////////////////////////////////
%% Initial parameters
%%

M = param.M;   %Number of neurons in the hidden layer
nit = param.nit;  %Number of iterations (epochs)

ntr = size(X_tr,1); %Number of training samples
d = size(X_tr,2); %Number of features

%Initialization of parameters
if (param.init == 1)
    w = param.w;
    w_2 = param.w_2;
else
    % Initialization of the weigths for the input layer
    w = randn(d+1,M).*0.1;% .*(10/sqrt((d+1)*M));
    % Initialization of the weigths for the output layer
    w_2 = randn(M+1,size(Y_tr,2)).*0.1; %.*(1/sqrt((size(Y_tr,2))*M));    
end

%Learning rate
alpha = param.alpha;  

%Activation function
act = param.act;

%Add the bias
X_trext = [X_tr ones(ntr,1)];   
T_train = Y_tr;

%//////////////////////////////////////////////////////////////////////////
%% Training
%%

minerr = 1e8;
wv = zeros(length(w(:)) + length(w_2(:)),nit);
gv = zeros(length(w(:)) + length(w_2(:)),nit);

%Sigmoid activation function in the hidden layer
if (act == 1)

    for k=1:nit
        %Training in batch
        
        %Forward propagation
        z = sigmoid(X_trext*w);
        z_ext = [z ones(size(z,1),1)];
        y = sigmoid(z_ext*w_2);

        %Backpropagation
        delta_output = (y - T_train);
        delta_input = w_2*delta_output'.*sigmoidGradient([X_trext*w ones(size(X_trext,1),1)]'); 
        delta_input = delta_input(1:end-1,:);
        grad_2 = z_ext'*delta_output./ntr;
        w_2 = w_2 - alpha.*grad_2;
        grad = X_trext'*delta_input'./ntr;
        w = w - alpha.*grad;

        %Re-evaluation of the error
        Z_tr = sigmoid(X_trext*w);
        Z_trext=[Z_tr ones(ntr,1)];
        y_tr = sigmoid(Z_trext*w_2);
        %Compute the error
        errtr(k) = mean(LRcost(y_tr,T_train)); 
%         %Save the weights if the training error is minimized
%         if (errtr(k) < minerr)
%             minerr = errtr(k);
%             w_save = w;
%             w_2_save = w_2;
%         end

        wv(:,k) = [w(:); w_2];
        gv(:,k) = [grad(:); grad_2];
    end
 
end


%Tanh activation function in the hidden layer
if (act == 2)

    for k=1:nit
        %Training in batch

        %Forward propagation
        z = tanh(X_trext*w);
        z_ext = [z ones(size(z,1),1)];
        y = sigmoid(z_ext*w_2);

        %Backpropagation
        delta_output = (y - T_train);
        delta_input = w_2*delta_output'.*tanhGradient([X_trext*w ones(size(X_trext,1),1)]'); 
        delta_input = delta_input(1:end-1,:);
        grad_2 = z_ext'*delta_output./ntr;
        w_2 = w_2 - alpha.*grad_2;
        grad = X_trext'*delta_input'./ntr;
        w = w - alpha.*grad;


        %Evaluación después del descenso por gradiente estocástico
        %Re-evaluation of the error
        Z_tr = tanh(X_trext*w);
        Z_trext=[Z_tr ones(ntr,1)];
        y_tr = sigmoid(Z_trext*w_2);
        %Compute the error
        errtr(k) = mean(LRcost(y_tr,T_train));
        %Save the weights if the training error is minimized
%         if (errtr(k) < minerr)
%             minerr = errtr(k);
%             w_save = w;
%             w_2_save = w_2;
%         end

        wv(:,k) = [w(:); w_2];
        gv(:,k) = [grad(:); grad_2];
    end
 
end


%Save the final weights
net.w = w;
net.w_2 = w_2;
net.act = act;

end


%Sigmoid function
function [o] = sigmoid(x)
o = 1./(1 + exp(-x));
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


