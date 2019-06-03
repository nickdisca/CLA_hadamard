function res=matvec_hadamard(UA,SA,VA,UB,SB,VB,x)

%step 1
W1=VB'*diag(x)*VA;

%step 2
W2=SB*W1*SA';

%step 3
res=diag(UB*W2*UA');

end