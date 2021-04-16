path = '/home/philipp/projects2/tomo/2019-09-09_kate_pd/'
s = readNPY([path '2019-10-05_merged_aligned_cropped.npy'])
s1 = permute(s, [2 1 3]);
%%

as = Anscombe_forward(s1);
v = as(:,:,25);
% w = den(:,:,25);
vx = v(1:60,1:60);
% imagesc(w)
vx = vx/max(vx(:));
sigma = sqrt(var(vx(:)));
% sigma = 0.01;
%%
den = zeros(size(as));
for i = 1:size(as,3)
%     i = 25;
    i
    print_to_screen = false;
    [PSNR, y_est] = BM3D(1, as(:,:,i), sigma, 'vn', print_to_screen);
    den(:,:,i) = y_est;
end
%%
% imagesc(y_est);
%%
diff = y_est-as(:,:,i);
imagesc(diff)
%%
figure(3);
ais1 = fftshift(fftshift(log10(abs(fft2(diff))),1),2);
ys = fftshift(fftshift(log10(abs(fft2(y_est))),1),2);
imagesc(ais1)
%%
stack_denoised = Anscombe_inverse_exact_unbiased(den);

%%
data = stack_denoised;
save([path '02_denoise/2019-10-07_bm3d_sigma0p1.m'],'data')