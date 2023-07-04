function dd = kfactor(Fobs,Fcalc)

% min|Fobs-k*Fcalc|^2
k = sum( Fobs(:).*conj(Fcalc(:))+conj(Fobs(:)).*Fcalc(:) )/sum( 2*conj(Fcalc(:)).*Fcalc(:) );
dd = real(k);

end