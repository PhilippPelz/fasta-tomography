function fit_result = stats_cal_PD(data,common_atoms2,common_atoms1,BoxSize1)
%  load common.mat
%  load  reconstruction_512_08_18.mat
%  data = Sraw;
%  BoxSize1 = 9;
  AA = (common_atoms1+common_atoms2)/2;
[box1coordinates, BoxCenter1, BoxRadius1, sphere1] = create_box(BoxSize1);
for num = 1:size(AA,2)
 
    DataBox1 = GetAtomBox(data,AA(:,num),BoxSize1);

    ymax_init       = DataBox1(BoxCenter1,BoxCenter1,BoxCenter1);
    fit_param_init  = [0,  ymax_init, 0,  0,  0,  0.2,  0.2,  0.8,  0.0,  0.0,  0.0];
    fixed           = [0,  0        ,  0,  0,  0,    0,    0,    0,    0,    0,    0];
    lb              = [0,  0        , -3, -3, -3,    0,    0,    0, -pi, 0, -pi];
    ub              = [inf,inf      ,  3,  3,  3,  inf,  inf,  inf, pi, pi,  pi];
 
    fit_result = fit_gauss3D_PD(fit_param_init, box1coordinates, DataBox1, fixed, lb, ub);
%     stats.fit{num} = fit_result;
%     stats.peak_bg(num) = fit_result(1);

end


% end