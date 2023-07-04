function dd = phi(r,atoms)

%Fe%
re = 2.481987;
A = 0.392811; B = 0.646243;
alpha = 9.818270; beta = 5.236411;
kappa = 0.170306; lambda = 0.340613;
dd1 = A*exp(-alpha*(r/re-1))./(1+(r/re-kappa).^20) - B*exp(-beta*(r/re-1))./(1+(r/re-lambda).^20);

%Pt%
re = 2.771916;
A = 0.556398; B = 0.696037;
alpha = 7.105782; beta = 3.789750;
kappa = 0.385255; lambda = 0.770510;
dd2 = A*exp(-alpha*(r/re-1))./(1+(r/re-kappa).^20) - B*exp(-beta*(r/re-1))./(1+(r/re-lambda).^20);

dd = (2-atoms).*dd1 + (atoms-1).*dd2;
