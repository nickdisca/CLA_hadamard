function res=matmat_hadamard(UA,SA,VA,UB,SB,VB,X)

res=zeros(size(UA,1),size(X,2));
for i=1:size(X,2)
    x=X(:,i);
    W1=VB'*diag(x)*VA;
    W2=SB*W1*SA';
    res(:,i)=diag(UB*W2*UA');
end

end