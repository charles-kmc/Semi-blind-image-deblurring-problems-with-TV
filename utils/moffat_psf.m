%%%  MOFFAT PSF
function fftcomplet_moffat = moffat_psf(im_shape, size, a, b)
    
%     size: integer representing the size of the kernel
%     beta and theta0: parameters of the point spread function
%     ouput: psf function
    
    center = (size+1)/2;
    X = -size + center:size - center;
    Y = -size + center:size - center;
    kernel = ones(size, size);
    for ii=1:size
        for jj=1:size
            b2 = b+2;
            xy = X(ii)^2 + Y(jj)^2;
            kernel(ii,jj) =  a^2 * (xy*a^2 / b + 1)^(-b2/2) / (2*pi) ;
        end
    end
    % Normalisation of the PSF
    kernel = kernel / sum(kernel(:));
    fftcomplet_moffat = resize(kernel, im_shape);

end



