function dd = EamPot_FePt2(model,atoms)

dd = 0;
rmin = 9.0;

rx = model(1,:); ry = model(2,:); rz = model(3,:);
M = size(rx,2);
for hh = 1:M
r = sqrt( (rx(hh)-rx).^2 + (ry(hh)-ry).^2 + (rz(hh)-rz).^2 );
r(hh) = inf;
ind = find(r<=rmin);
if isempty(ind)~=1
dd1 = sum( phi_ab(r(ind),atoms(hh),atoms(ind)) );
dd2 = sum( f(r(ind),atoms(ind)) );
dd  = dd + 0.5*dd1 + F_(dd2,atoms(hh));
end
end

