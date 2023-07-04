function dd = phi_ab(r,tp,atoms)

a = atoms; a(:) = tp; b = atoms;
dd = 0.5*( phi(r,a).*f(r,b)./f(r,a) + phi(r,b).*f(r,a)./f(r,b) );