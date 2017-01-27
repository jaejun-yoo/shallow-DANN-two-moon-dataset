% 20161221
% sDANN: Shallow Domain-Adversarial Training of Neural Networks (toy
% example)
% written by BISPL, KAIST, Jaejun Yoo
% e-mail: jaejun2004@gmail.com
% reference : https://arxiv.org/pdf/1505.07818v4.pdf

function [W,V,b,c] = sDANN(X, Y, X_adapt, learning_rate, hidden_layer_size, maxiter, lambda_adapt, adversarial_representation, seed)
% every annotation is done based on X: dim 200 by 2, hidden_layer_size = 15
% label Y should be 1 or 2
rng(seed)
[nb_examples, nb_features] = size(X); % X dim: 200 by 2 
nb_labels = length(unique(Y));
nb_examples_adapt = size(X_adapt,1);
% random initialization
W = randn(hidden_layer_size, nb_features); % dim: 15 by 2
V = randn(nb_labels,hidden_layer_size); % dim: 2 by 15
b = zeros(hidden_layer_size,1); % dim: 15 by 1
c = zeros(nb_labels,1); % dim: 2 by 1
U = zeros(hidden_layer_size,1); % dim: 15 by 1
d = 0;

best_valid_risk = 2;
continue_until = 30;
for t = 1:maxiter
    for i = 1:nb_examples
        x_t = X(i,:)'; % dim: 2 by 1
        y_t = Y(i); 
        
        hidden_layer = sigmoid(W*x_t + b); % dim: 15 by 1
        output_layer = softmax(V*hidden_layer + c); % dim: 2 by 1
        
        y_hot = zeros(nb_labels,1); % dim: 2 by 1
        y_hot(y_t) = 1; %one-hot vector
        
        delta_c = output_layer - y_hot; % dim: 2 by 1
        delta_V = delta_c * hidden_layer';% dim: 2 by 15
        delta_b = (V'*delta_c).*hidden_layer.*(1-hidden_layer); % dim: 15 by 1
        delta_W = delta_b*x_t'; % dim: 15 by 2
        
        if lambda_adapt == 0
            delta_U = 0;
            delta_d = 0;
        else
            % add domain adaptation regularizer from current domain
            gho_x_t = sigmoid((U'*hidden_layer) + d); % dim: 1 by 1
            
            delta_d = lambda_adapt * (1 - gho_x_t); % dim: 1 by 1
            delta_U = delta_d * hidden_layer; % dim:  15 by 1
            
            if adversarial_representation % true or false
                tmp = delta_d * U.* hidden_layer.*(1 - hidden_layer);
                delta_b = delta_b + tmp;
                delta_W = delta_W + tmp * x_t';
            end
            % add domain adaptation regularizer from other domain
            t_2 = randi(nb_examples_adapt);
            i_2 = t_2; % nb_examples_adapt
            x_t_2 = X_adapt(i_2, :)';
            hidden_layer_2 = sigmoid((W*x_t_2 + b));
            gho_x_t_2 = sigmoid((U'*hidden_layer_2) + d);
            
            delta_d = delta_d - (lambda_adapt * gho_x_t_2);
            delta_U = delta_U - (lambda_adapt * gho_x_t_2 * hidden_layer_2);
            
            if adversarial_representation
                tmp = -1 * lambda_adapt * gho_x_t_2 * U.* hidden_layer_2.* (1 - hidden_layer_2);
                delta_b = delta_b + tmp;
                delta_W = delta_W + tmp*x_t_2';
            end
        end
        W = W - (delta_W * learning_rate);
        b = b - (delta_b * learning_rate);
        
        V = V - (delta_V * learning_rate);
        c = c - (delta_c * learning_rate);
        
        
        U = U + (delta_U * learning_rate);
        d = d + (delta_d * learning_rate);
    end % END for i in range(nb_examples)

end% END for t in range(maxiter)

end
function y = sigmoid(z)
y = 1./(1+exp(-1*z));
end
function y = softmax(z)
y = exp(z)./sum(exp(z));
end
