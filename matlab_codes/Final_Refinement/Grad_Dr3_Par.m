function [scale dra] = Grad_Dr3_Par(Fobs,Fcalc,ra,ka,fa,bf,ht)

Fsub = (Fcalc-Fobs);

M = size(ra.rx,2);
drx = zeros(1,M); dry = zeros(1,M); drz = zeros(1,M);

s2 = ka.kx.^2+ka.ky.^2+ka.kz.^2;

fa_3 = fa(3,:);

parfor hh = 1:M

Cexp = -2*pi*i*ht(hh)*ka.kx.*fa_3.*exp( -2*pi*i*(ka.kx*ra.rx(hh)+ka.ky*ra.ry(hh)+ka.kz*ra.rz(hh))-bf(hh)*s2 ).*conj(Fsub);
drx(hh) = 2*real(sum(Cexp));

Cexp = -2*pi*i*ht(hh)*ka.ky.*fa_3.*exp( -2*pi*i*(ka.kx*ra.rx(hh)+ka.ky*ra.ry(hh)+ka.kz*ra.rz(hh))-bf(hh)*s2 ).*conj(Fsub);
dry(hh) = 2*real(sum(Cexp));

Cexp = -2*pi*i*ht(hh)*ka.kz.*fa_3.*exp( -2*pi*i*(ka.kx*ra.rx(hh)+ka.ky*ra.ry(hh)+ka.kz*ra.rz(hh))-bf(hh)*s2 ).*conj(Fsub);
drz(hh) = 2*real(sum(Cexp));

end

dra.drx = drx; dra.dry = dry; dra.drz = drz;

scale = sqrt( sum( (dra.drx).^2 + (dra.dry).^2 + (dra.drz).^2 ) );
dra.drx = dra.drx/scale; dra.dry = dra.dry/scale; dra.drz = dra.drz/scale;