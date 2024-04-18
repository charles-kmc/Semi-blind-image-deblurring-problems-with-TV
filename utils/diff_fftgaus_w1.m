% we are computing the gradient of the kernel
function [dkernel] = diff_fftgaus_w1(im_size, taille,w1,w2,phi)

   center = (taille + 1) / 2;
    %% x and y vector
    x = -taille + center: taille - center; 
    y = -taille + center: taille - center;
    
   %% Matrix of u and v
    [v,u] = ndgrid(x,y);
    %% transformation of x and y
    U =   u*cos(phi) - v * sin(phi);  
    V =   u*sin(phi) + v * cos(phi);
    
    %% sum of the psf and gardients
    [sum_psf, sum_d_w1,~] = Sum_gauss_psf(taille, w1, w2,phi);
    
    c = w1^2 * U.^2 + w2^2 * V.^2;
    
    f = ((w1*w2) / (2*pi)) * exp(-c/2);
    
    diff = (w2/(2*pi)) * (1 - w1^2*U.^2) .* exp(-c/2);
    
    dkernel = (diff * sum_psf - f * sum_d_w1)  / (sum_psf^2);
    dkernel = resize(dkernel, im_size);
end