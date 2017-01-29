% 20161221
% sDANN: Shallow Domain-Adversarial Training of Neural Networks (toy
% example)
% written by BISPL, KAIST, Jaejun Yoo
% e-mail: jaejun2004@gmail.com
% reference : https://arxiv.org/pdf/1505.07818v4.pdf

function []=sDANN_main()

clear all
% close all
clc
load('2Moons_v2.mat')

learning_rate = 0.05;
hidden_layer_size = 25;
lambda_adapt = 6;
maxiter = 800;
% adversarial_representation = true;
adversarial_representation = false;
seed = 2;
[W,V,b,c] = sDANN(X, Y, X_adapt, learning_rate, hidden_layer_size, maxiter, lambda_adapt, adversarial_representation, seed);

Y_adapt = predict(X_adapt,W,V,b,c);

figure,
plot(X(Y==1,1),X(Y==1,2),'bo')
hold on
plot(X(Y==2,1),X(Y==2,2),'b+')
plot(X_adapt(Y_adapt==1,1),X_adapt(Y_adapt==1,2),'ro')
plot(X_adapt(Y_adapt==2,1),X_adapt(Y_adapt==2,2),'r+')

figure,
subplot(311)
plot(X_adapt(Y_adapt==1,1),X_adapt(Y_adapt==1,2),'ro')
hold on
plot(X_adapt(Y_adapt==2,1),X_adapt(Y_adapt==2,2),'b+')
axis equal
subplot(312)
plot(X_adapt(Y_adapt==1,1),X_adapt(Y_adapt==1,2),'ro')
hold on 
plot(X_adapt(yt==1,1),X_adapt(yt==1,2),'gs')
axis equal
subplot(313)
plot(X_adapt(Y_adapt==2,1),X_adapt(Y_adapt==2,2),'b+')
hold on
plot(X_adapt(yt==-1,1),X_adapt(yt==-1,2),'gs')
axis equal
if adversarial_representation
   suptitle('predicted sDANN')
else
   suptitle('predicted sNN')
end

% figure, 
% plot(X_adapt(yt==1,1),X_adapt(yt==1,2),'ro')
% hold on
% plot(X_adapt(yt==-1,1),X_adapt(yt==-1,2),'r+')


end

function y = sigmoid(z)
y = 1./(1+exp(-1*z));
end
function y = softmax(z)
y = exp(z)./repmat(sum(exp(z)),2,1);
end
function output_layer=forward(X,W,V,b,c)
    hidden_layer = sigmoid(W*X'+ repmat(b,1,length(X))); % dim: 15 by 200
    output_layer = softmax(V*hidden_layer + repmat(c,1,length(X))); % dim: 2 by 200
end
function result = predict(X,W,V,b,c)
    output_layer = forward(X,W,V,b,c);
    [~, result] = max(output_layer,[],1); % dim: 1 by 200
end
