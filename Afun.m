function [y]=Afun(UA,SA,VA,UB,SB,VB,x,transp)

if ~transp %compute HAD'*HAD
    tmp=matvec_hadamard(UA,SA,VA,UB,SB,VB,x);
    y=matvec_hadamard(VA,SA',UA,VB,SB',UB,tmp);
else %compute HAD*HAD'
    tmp=matvec_hadamard(VA,SA',UA,VB,SB',UB,x);
    y=matvec_hadamard(UA,SA,VA,UB,SB,VB,tmp);
end

end