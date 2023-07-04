function img2 = FourierShift3D(img,dx,dy,dz)
%M. Bartels, UCLA, 2014
%based on code by R. Xu, UCLA
%shift 3D density matrix

[nx ny nz] = size(img);
[X Y Z] = ndgrid(-(nx-1)/2:(nx-1)/2,-(ny-1)/2:(ny-1)/2,-(nz-1)/2:(nz-1)/2);   %% an error was corrected here. L.

F = My_IFFTN(img);
Pfactor = exp(2*pi*1i*(dx*X/nx + dy*Y/ny + dz*Z/nz));
img2 = real(My_FFTN(F.*Pfactor));

end