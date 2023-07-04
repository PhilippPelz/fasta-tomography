function [DataBox, DataSphere]=GetAtomBox(DataMatrix,AtomPos,BoxSize)
%M. Bartels, UCLA, 2014
%get the density box and its spherical part of size BoxSize of a density matrix DataMatrix

BoxRadius = (BoxSize-1)/2;

[boxX,boxY,boxZ]=ndgrid(-BoxRadius:BoxRadius,-BoxRadius:BoxRadius,-BoxRadius:BoxRadius);

x=AtomPos(1);
y=AtomPos(2);
z=AtomPos(3);

%integer indices
fx=floor(x);fy=floor(y);fz=floor(z);

%off-center shift of peak
shiftx = (x-fx);shifty = (y-fy);shiftz = (z-fz);

if shiftx>0.5
    fx=fx+1;
    shiftx=shiftx-1;
end

if shifty>0.5
    fy=fy+1;
    shifty=shifty-1;
end

if shiftz>0.5
    fz=fz+1;
    shiftz=shiftz-1;
end

%calculate spherical mask
sphere = sqrt((boxX).^2+(boxY).^2+(boxZ).^2)<=BoxSize/2;

x_range = fx-BoxRadius-1:fx+BoxRadius+1;
y_range = fy-BoxRadius-1:fy+BoxRadius+1;
z_range = fz-BoxRadius-1:fz+BoxRadius+1;

DataBox = DataMatrix(x_range,y_range,z_range);

DataBox = FourierShift3D(DataBox,-shiftx,-shifty,-shiftz);

DataBox = DataBox (2:end-1,2:end-1,2:end-1);


DataSphere = DataBox.*sphere;
