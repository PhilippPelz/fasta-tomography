function dd = f(r,atoms)

%Fe%
fe = 1.885957; re = 2.481987;
beta = 5.236411; lambda = 0.340613;
dd1 = fe*exp(-beta*(r/re-1))./(1+(r/re-lambda).^20);

%Pt%
fe = 2.336509; re = 2.771916;
beta = 3.789750; lambda = 0.770510;
dd2 = fe*exp(-beta*(r/re-1))./(1+(r/re-lambda).^20);

dd = (2-atoms).*dd1 + (atoms-1).*dd2;
