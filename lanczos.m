function [T,Q]=lanczos(Afunc,v,maxit,tol)

q=v/norm(v);
Q=q;
alpha=[];
beta=[];

for k=1:maxit
    %compute w=A*q
    w=Afunc(q);
    
    %compute alpha
    alpha=[alpha; q'*w];
    
    %otrhogonalize vs cols of Q
    q=w-alpha(end)*q;
    if k>=2
        q=q-beta(end)*Q(:,end-1);
    end
    
    %reorthogonalize if necessary
    if norm(q)<0.7*norm(w)
        h=Q'*q;
        alpha(end)=alpha(end)+h(end); 
        q=q-Q*h;
    end
    
    %compute beta
    beta=[beta; norm(q)];
    
    %early stopping
    if beta(end)<tol
        break;
    end  

    %update q
    q=q/beta(end);
    
    %update Q
    Q=[Q q]; 
    
end

%correction
if k==maxit && beta(end)>=tol
    Q=Q(:,1:end-1);
end

%assemble T
T=diag(alpha)+diag(beta(1:end-1),-1)+diag(beta(1:end-1),1);

end