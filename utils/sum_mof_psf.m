function [sum_psf,sum_diff_alpha, sum_diff_beta] = sum_mof_psf(size, a, b)
center = (size+1)/2;
    X = -size + center:size - center;
    Y = -size + center:size - center;
    
    %% PSF sum
    kernel = ones(size, size);
    for ii=1:size
        for jj=1:size
            b2 =b+2;
            xii = X(ii);
            yjj = Y(jj);
            xy = xii^2 + yjj^2;
            kernel(ii,jj) =  a^2 * ((xy * a^2 )/ b + 1)^(-b2/2) / (2*pi);
        end
    end
    sum_psf = sum(kernel(:));
    
    %% sum of the differential of the PSF w.r.t alpha
    diffkernel_alpha = zeros(size, size);
    for ii=1:size
        for jj=1:size
            xy = X(ii)^2 + Y(jj)^2;
            diffkernel_alpha(ii,jj) = (2 - (((b+2)*xy*a^2) / (2*(b + xy*a^2)))) * (1 + xy*a^2 / b)^(-(b+2)/2) * (a/(2*pi));
        end
    end
    sum_diff_alpha = sum(diffkernel_alpha(:));
    
    %% sum of the differential of the PSF w.r.t beta
    diffkernel_beta = zeros(size, size);
    for ii=1:size
        for jj=1:size
            b2 = b+2;
            xy = X(ii)^2 + Y(jj)^2;
            cons1 = a^2 / (4*pi);
            diffkernel_beta(ii,jj) = (-log(xy*a^2/b + 1) + (b2*xy*a^2) / (b*(b + xy*a^2))) * (xy*a^2 / b + 1)^(-b2/2) * cons1;
        end
    end
    sum_diff_beta = sum(diffkernel_beta(:));
end
