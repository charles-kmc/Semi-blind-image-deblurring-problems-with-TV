
function psnr = PSNR(x,y)
    mse = 10*log10(max(x(:))^2);
    psnr = mse - 10*log10(norm(x(:) - y(:))^2 / numel(x));