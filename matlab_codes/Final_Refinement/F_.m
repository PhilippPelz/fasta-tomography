function dd = F_(rho_,tp)

if tp == 1
rhoe = 20.041463; rhos = 20.041463;
rhon = 0.85*rhoe;
rho0 = 1.15*rhoe;

Fn0 = -2.534992; Fn1 = -0.059605; Fn2 = 0.193065; Fn3 = -2.282322;
F0 = -2.54; F1 = 0; F2 = 0.200269; F3 = -0.148770;

eta = 0.391750;
Fe = -2.539945;

if rho_ < rhon
dd = Fn0 + Fn1*(rho_/rhon-1) + Fn2*(rho_/rhon-1)^2 + Fn3*(rho_/rhon-1)^3; 
end

if rhon <= rho_ && rho_ < rho0
dd = F0 + F1*(rho_/rhoe-1) + F2*(rho_/rhoe-1)^2 + F3*(rho_/rhoe-1)^3; 
end

if rho0 <= rho_
dd = Fe*(1-log((rho_/rhos)^eta))*(rho_/rhos)^eta;
end
return;
end

if tp == 2
rhoe = 33.367564; rhos = 35.205357;
rhon = 0.85*rhoe;
rho0 = 1.15*rhoe;

Fn0 = -1.455568; Fn1 = -2.149952; Fn2 = 0.528491; Fn3 = 1.222875;
F0 = -4.17; F1 = 0; F2 = 3.010561; F3 = -2.420128;

eta = 1.450000;
Fe = -4.145597;

if rho_ < rhon
dd = Fn0 + Fn1*(rho_/rhon-1) + Fn2*(rho_/rhon-1)^2 + Fn3*(rho_/rhon-1)^3; 
end

if rhon <= rho_ && rho_ < rho0
dd = F0 + F1*(rho_/rhoe-1) + F2*(rho_/rhoe-1)^2 + F3*(rho_/rhoe-1)^3; 
end

if rho0 <= rho_
dd = Fe*(1-log((rho_/rhos)^eta))*(rho_/rhos)^eta;
end
return;
end

