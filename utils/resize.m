function fftkernel = resize(kernel, im_shape)
    taille = length(kernel);
    x_shape = im_shape(1); 
    y_shape = im_shape(2);

    completkernel = zeros(x_shape, y_shape);
    
    completkernel(1: taille, 1: taille) = kernel;  
    
    %% fft2 
    fftkernel = fft2(completkernel);
end

