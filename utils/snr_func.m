function snr = snr_func(x,y)
    snr = 20 * log10(norm(x,2) / norm(x-y,2));
end