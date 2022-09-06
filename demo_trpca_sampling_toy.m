addpath(genpath(cd))
clear all
close all

%%
n1 = 1000;
n2 = n1;
n3 = 3;
r = 0.05*n1;  
L1 = randn(n1,r,n3)/n1;
L2 = randn(r,n2,n3)/n2;
L = tprod(L1,L2); 
rho = 0.05;
m = rho*n1*n2*n3;
temp = rand(n1*n2*n3,1);
[B,I] = sort(temp);
I = I(1:m);
Omega = zeros(n1,n2,n3);
Omega(I) = 1;
E = sign(rand(n1,n2,n3)-0.5);
S = Omega.*E; 
D = L+S;

%% 列空间学习
p1=0.2; 
m1=p1*n2; 
randnumber=randperm(n2);
I=randnumber(1:m1);
DS1=D(:,I,:);
opts.tol = 1e-8;
opts.mu = 1e-4;
opts.rho = 1.1;
opts.DEBUG = 1;
lambda = 1/sqrt(n3*max(n1,n2));
tic
[Lhat,~] = trpca_tnn(DS1,lambda,opts);
t1 = toc;
[Uhat,Sigma,VS1_T]=tsvd(Lhat,'skinny');
[U,SIGMA,V_T]=tsvd(L,'skinny');

%% 行空间学习
p2=0.1;
m2=p2*n1;
randnumber=randperm(n1);
I=randnumber(1:m2);
S2Uhat=Uhat(I,:,:);
S2D=D(I,:,:);
tic
Qhat=min_L1(S2Uhat,S2D);
t2 = toc;
Lhat=tprod(Uhat,Qhat);
Shat=D-Lhat;

Lr = norm(L(:)-Lhat(:))/norm(L(:))
Sr = norm(S(:)-Shat(:))/norm(S(:))
t = t1+t2





