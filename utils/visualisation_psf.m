w11 = 0.4; 
w12 = 0.3;
w2 = 0.3;

size = 7;
%% spatial domain
psf1 = Gaussian_psf(size, w11, w2,0);

psf2 = Gaussian_psf(size, w12,w2,0);



%% 3D domain
cen = (size+1)/2;



%% coupe in spatial and Fourier domain
f1c = figure;
plot(psf1(cen,:),'LineWidth',1.5,'MarkerSize',10);hold on;
plot(psf2(cen,:),'LineWidth',1.5,'MarkerSize',10);
legend('w11 =0,4 ', 'w12 = 0.25');title(strcat('p1 =',num2str(sum(psf1(:))), 'p2=', num2str(sum(psf2(:)))));
% saveas(f1c,['./PSF_PLOTS/psf_center.png']);
