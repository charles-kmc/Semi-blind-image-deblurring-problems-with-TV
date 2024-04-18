function complet_diffkernel = diff_moffat_alpha(im_shape, size, a, b)

%     size: integer representing the size of the kernel
%     beta and theta0: parameters of the point spread function
%     ouput: psf function
%     alpha = 1 / alpha; 
    [sum_Psf,sum_diff_alpha, ~] = sum_mof_psf(size, a, b);
    center = (size+1)/2;
    X = -size + center:size - center;
    Y = -size + center:size - center;
    diffkernel = zeros(size, size);
    for ii=1:size
        for jj=1:size
            b2 = b+2;
            xy = X(ii)^2 + Y(jj)^2;
            f = a^2 * ((xy * a^2 )/ b + 1)^(-b2/2) / (2*pi);
            diff = (2 - (((b+2)*xy*a^2) / (2*(b + xy*a^2)))) * (1 + xy*a^2 / b)^(-(b+2)/2) * (a/(2*pi));
            diffkernel(ii,jj) = (diff * sum_Psf - f * sum_diff_alpha) / (sum_Psf)^2;
        end
    end
    complet_diffkernel = resize(diffkernel, im_shape);
end