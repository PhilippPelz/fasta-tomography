function [scale dra] = EamGrad_FePt2(model,atoms)

delta_r = 0.005;
rmin = 9.0;

rx = model(1,:); ry = model(2,:); rz = model(3,:);
drx = rx; dry = ry; drz = rz;
M = size(rx,2);

for hh = 1:M
ff0 = 0;
r = sqrt( (rx(hh)-rx).^2 + (ry(hh)-ry).^2 + (rz(hh)-rz).^2 );
r(hh) = inf;
ind = find(r<=rmin);
if isempty(ind)~=1
dd1 = sum( phi_ab(r(ind),atoms(hh),atoms(ind)) );
dd2 = sum( f(r(ind),atoms(ind)) );
ff0  = 0.5*dd1 + F_(dd2,atoms(hh));
end

ff1 = 0;
r = sqrt( (rx(hh)+delta_r-rx).^2 + (ry(hh)-ry).^2 + (rz(hh)-rz).^2 );
r(hh) = inf;
if isempty(ind)~=1
dd1 = sum( phi_ab(r(ind),atoms(hh),atoms(ind)) );
dd2 = sum( f(r(ind),atoms(ind)) );
ff1  = 0.5*dd1 + F_(dd2,atoms(hh));
end

ff2 = 0;
r = sqrt( (rx(hh)-rx).^2 + (ry(hh)+delta_r-ry).^2 + (rz(hh)-rz).^2 );
r(hh) = inf;
if isempty(ind)~=1
dd1 = sum( phi_ab(r(ind),atoms(hh),atoms(ind)) );
dd2 = sum( f(r(ind),atoms(ind)) );
ff2  = 0.5*dd1 + F_(dd2,atoms(hh));
end

ff3 = 0;
r = sqrt( (rx(hh)-rx).^2 + (ry(hh)-ry).^2 + (rz(hh)+delta_r-rz).^2 );
r(hh) = inf;
if isempty(ind)~=1
dd1 = sum( phi_ab(r(ind),atoms(hh),atoms(ind)) );
dd2 = sum( f(r(ind),atoms(ind)) );
ff3  = 0.5*dd1 + F_(dd2,atoms(hh));
end

drx(hh) = (ff1 - ff0)/delta_r;
dry(hh) = (ff2 - ff0)/delta_r;
drz(hh) = (ff3 - ff0)/delta_r;
end

scale = sqrt( sum( drx.^2 + dry.^2 + drz.^2 ) );
drx = drx/scale; dry = dry/scale; drz = drz/scale;
dra.drx = drx; dra.dry = dry; dra.drz = drz;