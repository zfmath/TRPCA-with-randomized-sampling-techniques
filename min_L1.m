function x=min_L1(A,b)
% The ADMM for the l1 norm min
% 
% min_{x} ||b-Ax||_1
%
%min||u||_{1} s.t. b-Ax=u
%
%L=argmin{||u||_{1}+y^{T}(Ax+u-b)+eta/2||Ax+u-b||_{F}^{2}}
%
%

%Initialize
tol = 1e-8; 
max_iter = 200;
rho = 1.1;
mu=1e-4;
max_mu = 1e10;


[n1,n2,n3] = size(b);
[~,r,~]=size(A);
x=zeros(r,n2,n3);
u=zeros(n1,n2,n3);
y=zeros(n1,n2,n3);

for iter = 1 : max_iter
lastu=u;
lastx=x;
%Update u

u=prox_l1(b-tprod(A,x)-y/mu,1/mu);
 
%Update x

x=tprod(tinv(tprod(tran(A),A)),tprod(tran(A),b-y/mu-u));

dz = b-tprod(A,x)-u;
chgu = max(abs(lastu(:)-u(:)));
chgx = max(abs(lastx(:)-x(:)));
chg = max([chgu chgx max(abs(dz(:)))]);

    if chg < tol
        break;
    end 
    
%Update y
y=y+mu*(tprod(A,x)+u-b);

%Update eta
mu=min(rho*mu,max_mu);    
disp(['iter ' num2str(iter) ]);
end


