function fftkernel = FFT(x)
    fftkernel = fft2(x);   
%     fftkernel(abs(fftkernel)<1e-6) = 1e-6;
end