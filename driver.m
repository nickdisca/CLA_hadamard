%% initialize driver
clear all; colors=get(gca,'ColorOrder'); close all; clc;
set(0,'defaultAxesFontSize',20); set(0,'defaultLineLineWidth',2);

%% test using matrices created by hand
m=8; n=5;
UA=reshape(1:m*2,m,2)/100; SA=[1 2; 3 4]; VA=reshape(13+(1:n*2),n,2)/100; A=UA*SA*VA';
UB=reshape(-7+(1:m*3),m,3)/100; SB=[1 3 2; 6 5 8; 9 7 4]; VB=reshape(-42+(1:n*3),n,3)/100; B=UB*SB*VB';
HAD=A.*B;
x=(1:n)';
xx=(1:m)';

value=matvec_hadamard(UA,SA,VA,UB,SB,VB,x);
value_ex=HAD*x;

value=matvec_hadamard(VA,SA',UA,VB,SB',UB,xx);
value_ex=HAD'*xx;

%% find the reduced SVD of hadamard product and check with Matlab
maxit=1000;
tol=1e-8;
x0=randn(size(UA,1),1);
xx0=randn(size(VA,1),1);
Afunc=@(x,transp) Afun(UA,SA,VA,UB,SB,VB,x,transp);

%use HAD*HAD' to compute left singular vectors 
[T,Q]=lanczos(@(x) Afunc(x,true),x0,maxit,tol);
[P,LAM]=eig(T); 
[~,ids]=sort(diag(LAM),'descend'); LAM=LAM(ids,ids); P=P(:,ids);
S1=diag(sqrt(diag(LAM)));
U1=Q*P;

%retrieve right singular vectors
V1=matmat_hadamard(VA,SA',UA,VB,SB',UB,U1/S1');

%use HAD'*HAD to compute right singular vectors
[T,Q]=lanczos(@(x) Afunc(x,false),xx0,maxit,tol);
[P,LAM]=eig(T); 
[~,ids]=sort(diag(LAM),'descend'); LAM=LAM(ids,ids); P=P(:,ids);
S2=diag(sqrt(diag(LAM)));
V2=Q*P;

%retrieve left singular vectors
U2=matmat_hadamard(UA,SA,VA,UB,SB,VB,V2/S2);

%check with Matlab svd computing the first k singular values
[U_ex,S_ex,V_ex]=svds(HAD,size(P,1));

S_ex
S1
S2

U_ex
U1
U2

V_ex
V1
V2

%% function evaluation

%create matrices A,B,C
f=@(x,y) 1./(x+y);
g=@(x,y) 1./sqrt(x.^2+y.^2);
epsilon=1e-4;
maxit=1000;
tol=epsilon^2;

M=1; N=2;
v1=0.1:0.1:M;
v2=0.1:0.1:N;

[X,Y]=meshgrid(v1,v2);
A=f(X',Y');
B=g(X',Y');
C=A.*B;

%compute svd for A to get the low-rank approximation
[U,S,V]=svd(A);
ids=(diag(S)>=epsilon);
SA=S(ids,ids);
UA=U(:,1:size(SA,1));
VA=V(:,1:size(SA,2));

%compute svd for B to get the low-rank approximation
[U,S,V]=svd(B);
ids=(diag(S)>=epsilon);
SB=S(ids,ids);
UB=U(:,1:size(SB,1));
VB=V(:,1:size(SB,2));

%get the low-rank approximation of C using lanczos (use as a test HAD*HAD' and HAD'*HAD)
x0=randn(size(UA,1),1);
Afunc=@(x,transp) Afun(UA,SA,VA,UB,SB,VB,x,transp);
[T,Q]=lanczos(@(x) Afunc(x,true),x0,maxit,tol);
[P,LAM]=eig(T); 
[~,ids]=sort(diag(LAM),'descend'); LAM=LAM(ids,ids); P=P(:,ids);
SC1=diag(sqrt(diag(LAM)));
UC1=Q*P;
VC1=matmat_hadamard(VA,SA',UA,VB,SB',UB,UC1/SC1');

x0=randn(size(VA,1),1);
[T,Q]=lanczos(@(x) Afunc(x,false),x0,maxit,tol);
[P,LAM]=eig(T); 
[~,ids]=sort(diag(LAM),'descend'); LAM=LAM(ids,ids); P=P(:,ids);
SC2=diag(sqrt(diag(LAM)));
VC2=Q*P;
UC2=matmat_hadamard(UA,SA,VA,UB,SB,VB,VC2/SC2);

%check with Matlab implementation
[U,S,V]=svd(C);
ids=(diag(S)>=epsilon);
S=S(ids,ids);
U=U(:,1:size(S,1));
V=V(:,1:size(S,2));

%compute errors
error_lanczos=norm((UA*SA*VA').*(UB*SB*VB')-UC1*SC1*VC1','fro')

%compare dimensions
rank_Lanczos=size(SC1,1)
LR_representation=size(SA,1)*size(SB,1)

SC1
SC2
S

UC1
UC2
U

VC1
VC2
V

%% performance evaluation
clear all; colors=get(gca,'ColorOrder'); close all; clc;
set(0,'defaultAxesFontSize',20); set(0,'defaultLineLineWidth',2);

f=@(x,y) 1./(x+y);
g=@(x,y) 1./sqrt(x.^2+y.^2);
M_vec=50:50:300; N_vec=M_vec;
epsilon=1e-4;
maxit=1000;
tol=epsilon^2;

time_lanczos_1=zeros(size(M_vec));
time_lanczos_2=zeros(size(M_vec));
time_svd=zeros(size(M_vec));
error_lanczos_1=zeros(size(M_vec));
error_lanczos_2=zeros(size(M_vec));
error_svd=zeros(size(M_vec));
dim_lanczos_1=zeros(size(M_vec));
dim_lanczos_2=zeros(size(M_vec));
dim_svd=zeros(size(M_vec));
dim_lr_prod=zeros(size(M_vec));


for i=1:length(M_vec)
    
    fprintf('Iteration %d/%d',i,length(M_vec));

    %create matrices A,B,C
    v1=0.1:0.1:M_vec(i); v2=0.1:0.1:N_vec(i);
    [X,Y]=meshgrid(v1,v2);
    A=f(X',Y');
    B=g(X',Y');
    C=A.*B;
    
    %compute svd for A to get the low-rank approximation
    [U,S,V]=svd(A);
    ids=(diag(S)>=epsilon);
    SA=S(ids,ids);
    UA=U(:,1:size(SA,1));
    VA=V(:,1:size(SA,2));

    %compute svd for B to get the low-rank approximation
    [U,S,V]=svd(B);
    ids=(diag(S)>=epsilon);
    SB=S(ids,ids);
    UB=U(:,1:size(SB,1));
    VB=V(:,1:size(SB,2));
    
    fprintf('...assembly');

    %get the low-rank approximation of C using lanczos on HAD*HAD'
    start=tic;
    
    x0=randn(size(UA,1),1);
    Afunc=@(x,transp) Afun(UA,SA,VA,UB,SB,VB,x,transp);
    [T,Q]=lanczos(@(x) Afunc(x,true),x0,maxit,tol);
    [P,LAM]=eig(T);
    [~,ids]=sort(diag(LAM),'descend'); LAM=LAM(ids,ids); P=P(:,ids);
    SC1=diag(sqrt(diag(LAM)));
    UC1=Q*P;
    VC1=matmat_hadamard(VA,SA',UA,VB,SB',UB,UC1/SC1');
    
    time_lanczos_1(i)=toc(start); 
    
    fprintf('...Lanczos1');
    
    %get the low-rank approximation of C using lanczos on HAD'*HAD
    start=tic;
    
    x0=randn(size(VA,1),1);
    [T,Q]=lanczos(@(x) Afunc(x,false),x0,maxit,tol);
    [P,LAM]=eig(T);
    [~,ids]=sort(diag(LAM),'descend'); LAM=LAM(ids,ids); P=P(:,ids);
    SC2=diag(sqrt(diag(LAM)));
    VC2=Q*P;
    UC2=matmat_hadamard(UA,SA,VA,UB,SB,VB,VC2/SC2);
    
    time_lanczos_2(i)=toc(start);
    
    fprintf('...Lanczos2');
    
    %run SVD on C
    start=tic;
        
    [U,S,V]=svd(C);
    ids=(diag(S)>=epsilon);
    S=S(ids,ids);
    U=U(:,1:size(S,1));
    V=V(:,1:size(S,2));

    time_svd(i)=toc(start);
    
    fprintf('...SVD');
    
    %compute errors
    %error_lanczos_1(i)=norm(C-UC1*SC1*VC1','fro');
    %error_lanczos_2(i)=norm(C-UC2*SC2*VC2','fro');
    error_lanczos_1(i)=norm((UA*SA*VA').*(UB*SB*VB')-UC1*SC1*VC1','fro');
    error_lanczos_2(i)=norm((UA*SA*VA').*(UB*SB*VB')-UC2*SC2*VC2','fro');
    error_svd(i)=norm(C-U*S*V','fro');
    
    %compare dimensions (using Lanczos 1)
    dim_lanczos_1(i)=size(SC1,1);
    dim_lanczos_2(i)=size(SC2,1);
    dim_svd(i)=size(S,1);
    dim_lr_prod(i)=size(SA,1)*size(SB,1);
    
    fprintf('...End\n');

end

figure;
semilogy(M_vec,time_svd,'color',colors(1,:));
hold on;
semilogy(M_vec,time_lanczos_1,'color',colors(2,:));
semilogy(M_vec,time_lanczos_2,'color',colors(3,:));
xlabel({'$M,N$'},'interpreter','latex'); ylabel({'time [s]'},'interpreter','latex'); 
legend({'Standard svd','Lanczos 1','Lanczos 2'},'interpreter','latex','location','southeast');

figure;
semilogy(M_vec,error_svd,'color',colors(1,:));
hold on;
semilogy(M_vec,error_lanczos_1,'color',colors(2,:));
semilogy(M_vec,error_lanczos_2,'color',colors(3,:));
xlabel({'$M,N$'},'interpreter','latex'); ylabel({'Error'},'interpreter','latex'); 
legend({'Standard svd','Lanczos 1','Lanczos 2'},'interpreter','latex','location','southwest');

figure;
plot(M_vec,dim_lr_prod,'color',[0 0 0]);
hold on;
plot(M_vec,dim_svd,'color',colors(1,:));
plot(M_vec,dim_lanczos_1,'color',colors(2,:));
plot(M_vec,dim_lanczos_2,'color',colors(3,:));
xlabel({'$M,N$'},'interpreter','latex'); ylabel({'Dimensions'},'interpreter','latex'); 
legend({'$k_Ak_B$','$k$ (svd)', '$k$ (Lanczos 1)','$k$ (Lanczos 2)'},'interpreter','latex','location','northwest');

figure;
plot(M_vec,dim_svd,'color',colors(1,:));
hold on;
plot(M_vec,dim_lanczos_1,'color',colors(2,:));
plot(M_vec,dim_lanczos_2,'color',colors(3,:));
xlabel({'$M,N$'},'interpreter','latex'); ylabel({'Dimensions'},'interpreter','latex'); 
legend({'$k_Ak_B$','$k$ (svd)', '$k$ (Lanczos 1)','$k$ (Lanczos 2)'},'interpreter','latex','location','northwest');