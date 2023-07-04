function M = find_maxima3D(mat3d,mask)
%M. Bartels, UCLA, 2014
%find all maxima of a 3D matrix
%with a local neighbourhood defined by mask

s = size(mask,1);
c = round((s+1)/2);
mask=mask>0;
mask(c,c,c) = false; %exclude the pixel itself from the neighbourhood

%for each voxel, assign the maximum in the neighbourhood
mat3d_neighbours = imdilate(mat3d,mask);

%a maximum is where the matrix is larger than all neighbors
M = mat3d > mat3d_neighbours; 

end