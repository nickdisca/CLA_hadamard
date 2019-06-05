%% initialize playground (plot singular values)
clear all; colors=get(gca,'ColorOrder'); close all; clc;
set(0,'defaultAxesFontSize',20); set(0,'defaultLineLineWidth',2);

%create matrices A,B,C
f=@(x,y) 1./(x+y);
g=@(x,y) 1./sqrt(x.^2+y.^2);
epsilon=1e-4;
maxit=1000;
tol=epsilon^2;

M=50; N=50;
v1=0.1:0.1:M;
v2=0.1:0.1:N;

[X,Y]=meshgrid(v1,v2);
A=f(X,Y);
B=g(X,Y);
C=A.*B;

[~,SC,~]=svd(C);
nb_valsC=min(size(SC));
[~,SA,~]=svd(A);
nb_valsA=min(size(SA));
[~,SB,~]=svd(B);
nb_valsB=min(size(SB));
figure;
semilogy(1:nb_valsA,sort(diag(SA),'descend'),'color',colors(1,:)); 
hold on;
semilogy(1:nb_valsB,sort(diag(SB),'descend'),'color',colors(2,:));
semilogy(1:nb_valsC,sort(diag(SC),'descend'),'color',colors(3,:));
xlabel({'$i$'},'interpreter','latex'); ylabel({'$\sigma_i$'},'interpreter','latex'); 
legend({'$A$','$B$','$C$'},'interpreter','latex','location','northeast');

%% initialize playground 2 (check orthogonality)
clear all; colors=get(gca,'ColorOrder'); close all; clc;
set(0,'defaultAxesFontSize',20); set(0,'defaultLineLineWidth',2);

f=@(x,y) 1./(x+y);
g=@(x,y) 1./sqrt(x.^2+y.^2);
epsilon=1e-4;
maxit=1000;
tol=epsilon^2;

M=5; N=5;
v1=0.1:0.1:M;
v2=0.1:0.1:N;

[X,Y]=meshgrid(v1,v2);
A=f(X,Y);
B=g(X,Y);
C=A.*B;

[U,S,V]=svd(A); ids=(diag(S)>=epsilon); SA=S(ids,ids); UA=U(:,1:size(SA,1)); VA=V(:,1:size(SA,2));
UA'*UA

[U,S,V]=svd(B); ids=(diag(S)>=epsilon); SB=S(ids,ids); UB=U(:,1:size(SB,1)); VB=V(:,1:size(SB,2));
UB'*UB

for i=1:size(A,1), KRT(i,:)=kron(UA(i,:),UB(i,:)); end
for j=1:size(A,2), KR(:,j)=kron(VA(j,:),VB(j,:)); end
SIG=kron(SA,SB);
norm((UA*SA*VA').*(UB*SB*VB')-KRT*SIG*KR)