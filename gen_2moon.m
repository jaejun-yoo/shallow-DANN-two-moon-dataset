% 20161221
% two-moon dataset generator
% sDANN: Shallow Domain-Adversarial Training of Neural Networks (toy
% example)
% written by BISPL, KAIST, Jaejun Yoo
% e-mail: jaejun2004@gmail.com

center1 = -1.5;
center2 = 1.5;
nb_sample = 1000;
radius = 2;
theta = linspace(0,pi,nb_sample);
noise = rand(1,nb_sample)*0.6;
semi_up = [(radius+noise).*cos(theta) + center1;(radius-1+noise).*sin(theta)-0.2];
semi_down = [(radius+noise).*cos(-1*theta) + center2; (radius+noise).*sin(-1*theta)];
figure;
plot(semi_up(1,:),semi_up(2,:),'bo')
hold on
plot(semi_down(1,:),semi_down(2,:),'ro')

x = [semi_up,semi_down]';
y = [ones(length(semi_up),1);-1*ones(length(semi_down),1)];
idx = randperm(2*nb_sample,2*nb_sample);
xt = x(idx(1:nb_sample),:);
yt = y(idx(1:nb_sample));
x = x(idx(nb_sample+1:end),:);
y = y(idx(nb_sample+1:end));

theta = 15;
theta = theta*pi/180;
rotation = [  [cos(theta), sin(theta)]; [-sin(theta), cos(theta)] ];
xt = xt * rotation;

figure,
plot(x(y==1,1),x(y==1,2),'bo')
hold on 
plot(x(y==-1,1),x(y==-1,2),'b+')
plot(xt(yt==1,1),xt(yt==1,2),'ro')
plot(xt(yt==-1,1),xt(yt==-1,2),'r+')
axis equal

save('2Moons_v2','x','xt','y','yt');



