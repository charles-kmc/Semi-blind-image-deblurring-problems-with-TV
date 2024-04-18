function [xMAP] = myula(op, im)
gradF = op.gradF;
x0 = op.y;
lambda = op.lambda;
gamma = op.gamma;
theta = op.theta_op;
sample = op.samples;
tau = op.tau_op;
proxG = op.proxG;

x_shape = size(im);
x = x0;
fprintf('|=======> Running myula with theta: %4d and tau: %3d', theta, tau);
for ii=2:1:sample-1
    z = randn(x_shape);
    prox = proxG(x, lambda , theta);
    x = (1 - gamma ./ lambda) .* x - gamma .* (gradF(x, tau) - prox ./ lambda) + sqrt(2 .* gamma) *z;
    %Display
    if mod(ii, round(sample / 100)) == 0
            fprintf('\b\b\b\b%2d%%\n', round(ii / (sample) * 100));
    end
end
xMAP = x;

    