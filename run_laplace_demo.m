% Charlesquin Kemajou Mbakam: mbakamkemajou@gmail.org
% Heriot-Watt University
% date: Juin 2022

%% Experiment - Deblurring with Total-Variation prior: Laplace point spread function as kernel.

% -------------------------------------------------------------------------%
% -------------------------------------------------------------------------%

clear all;clc;

%% Create array with test images names
  addpath([pwd,'/utils']);
  addpath([pwd,'/SALSA']);
  addpath([pwd,'/SAPG']);

% test images
  testImg = dir('images');
  testImg = {testImg(3:end).name};

%Check that tesexitt images are accesible
  if sum(size(testImg)) == 0
      error('No images found, please check that the images directory is in the MATLAB path');
  end
%Check that SALSA solver is in the MATLAB path
  if exist('SALSA_v2')==0
      error('SALSA package not found. Please make sure that the salsa solver is added to your MATLAB path.  The code can be obtained from http://cascais.lx.it.pt/~mafonso/salsa.html');
    end
% Set to true to save additional plots
  save_logpi_plots= true; % to check stabilisation of the MYULA sampler
  save_gx_plot=true; % to check evolution of the regulariser g and the gradient approx

%% Parameter Setup
    op.samples = 20000; % max iterations for SAPG algorithm to estimate theta
    op.stopTol = 1e-5; % tolerance in relative change of theta_EB to stop the algorithm
    op.burnIn = (op.samples * 80) / 100;% iterations we ignore before taking the average over iterates theta_n
    op.warm_sample = 1;
    op.warmup = 15000; % number of warm-up iterations with fixed theta for MYULA sampler
    op.lambdaMax = 0.1;% max smoothing parameter for MYULA
    op.gammaFrac=0.98; % we set gamma=op.gammaFrac*gamma_max

  % theta: regularisation paramter
    op.min_th = 1e-3; %
    op.max_th = 1; %
    op.th_init = 0.01; % theta_0 initialisation of the SAPG algorithm

  % parameter of the Laplace kernel
    op.min_b = 1e-3;
    op.max_b = 1;
    op.b_init = 0.1; % b_0 initialization of the SAPG algorithm

  % Noise variance
    op.BSNR_max = 45;
    op.BSNR_min = 15;

  % PSF paameters
    psf_size= 7;
    op.psf_size = psf_size;

  % parameters of the SAPG
    op.d_exp =  0.8;
    op.d_scale =  0.01/op.th_init;

    randn('state',1); % Set rnd generators (for repeatability)

%%
    all = 0;
    fix_b = 0;
    fix_sigma = 0;
    op.fix_b = fix_b;
    op.fix_sigma = fix_sigma;

%% RUN EXPERIMENTS
for b = [0.3]
    op.b = b;
    if fix_b
        fprintf('\n\n\t\t Fix b \n\n');
        op.b_init = b;
    end
    for snr = [30]
        op.BSNR = snr;
        fprintf('\n\n===========<< SNR  = %d  \t|\t B  = %d>>============\n\n', snr, b);
        for i_im = [8]
            filename=testImg{i_im};
            [~,fix,~] = fileparts(char(filename));
            name = fix;
            %% Generate a noisy and blurred observation vector 'y'
            %  original image 'x'
              fprintf(['File:' filename '\n']);
              x = double(imread(strcat('images/', filename)));
              op.x = x;
              dimX = numel(x);
              im_size = size(x);

            % kernel function
              H_FFT  = @(b) laplace_psf(im_size, psf_size, b);

            % conjugate of the PSF
              HC_FFT = @(b) conj(H_FFT(b));

            % differential w.r.t b
              diff_fftkernel_b = @(b) diff_laplace_b(im_size, psf_size, b);

            %% psf and its gradients
              A = @(x, b) real(ifft2(H_FFT(b) .* fft2(x)));
              AT = @(x, b) real(ifft2(HC_FFT(b) .* fft2(x)));
              diff_A_b = @(x, b) real(ifft2(diff_fftkernel_b(b) .* fft2(x)));

            %% Max eigen value of A through iterative method
              evMax=max_eigenval_Laplace(A,AT, 1, im_size, 1e-4, 1e4, 0);
              fprintf('evmax=%d\n',evMax);

            %% observation y
              Ax =  real(A(x, b));
              sigma =  norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(op.BSNR/10));
              fprintf('Sigma^2 = %d\t\t snr:%d\n\n', sigma^2, op.BSNR);
              op.sigma = sigma;
              sigma_min= norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(op.BSNR_min/10));
              sigma_max = norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(op.BSNR_max/10));
              op.sigma_init = (sigma_min^2 + sigma_max^2) / 2;
              if fix_sigma
                  op.sigma_init = sigma^2;
              end
              op.sigma_min = sigma_min^2;
              op.sigma_max = sigma_max^2;
              y = Ax + sigma * randn(size(Ax));
              op.X0 = y;

            %% Experiment setup: functions related to Bayesian model

            % Likelihood and gradients
              op.f = @(x, b, sigma2) (norm(y-A(x, b),'fro')^2)/(2*sigma2);
              op.gradF = @(x, b, sigma2) real(AT(A(x,b) - y, b)/sigma2);
              op.grad_b = @(x, b, sigma2) real(sum(sum(diff_A_b(x,b) .* (A(x, b) - y))) / sigma2);
              op.gradF_sigma = @(x, b, sigma2) ((norm(y - A(x, b) ,'fro')^2)/(2*sigma2^2) - dimX/(2*sigma2)) ;
              Lf = @ (sigma2) evMax^2 / sigma2;
              op.Lf = max(Lf(sigma_min^2), Lf(sigma_max^2));

            % Set algorithm parameters that depend on Lf
              op.lambda=   min(5/op.Lf,op.lambdaMax);%smoothing parameter for MYULA sampler
              op.gamma_max = 1 / (op.Lf+(1/op.lambda));%Lf is the lipschitz constant of the gradient of the smooth part
              op.gamma= 10 * op.gammaFrac * op.gamma_max;%discretisation step MYULA

            % Regulariser % TV norm
              op.g = @(x) TVnorm(x); %g(x) for TV regularizer
              chambolleit = 25;

            % Proximal operator of g(x)
              op.proxG = @(x,lambda,theta) chambolle_prox_TV_stop(x,'lambda',lambda*theta,'maxiter',chambolleit);
              Psi = @(x,th) op.proxG(x, 1, th); % define this format for SALSA solver

            % We use this scalar summary to monitor convergence
              op.logPi = @(x,theta, b, sigma2) -op.f(x, b, sigma2) -theta*op.g(x);

            %% Run SAPG Algorithm 1 to compute theta_EB
              tic;
              [theta_EB, b_EB, sigma2_EB, results] = SAPG_algorithm_laplace(y, op);
              SAPG_time = toc;
              fprintf('\n\n\t\t=========>>> SAPG Elapsed: %d <<<=========\n\n', SAPG_time);
              results.SAPG_time = SAPG_time;
              results.theta_EB = theta_EB;
              results.b_EB = b_EB;
              results.sigma2_EB = sigma2_EB;
              results.b = b;
              sigma_EB = sigma2_EB;
              results.sigma = sigma;
              last_b = results.last_b;
              results.op = op;
              results.x=x;
              results.y=y;
              fprintf('Theta EB: %d\t|\t B EB: %d\t|\t Last b: %d\t|\t Sigma EB: %d', theta_EB, b_EB, last_b, sigma_EB);

            %% Solve MYULA problem with theta_EB, b_EB, mu_EB and sigma_EB

            % b_EB, sigma_EB, theta_EB,
              tic;
              global calls;
              calls = 0;
              outeriters = 500;
              A1 = @(x) A(x, b_EB);
              AT1 = @(x) AT(x, b_EB);
              A1 = @(x) callcounter(A1,x);
              AT1 = @(x) callcounter(AT1,x);
              inneriters = 1;
              tol = 1e-5; %taken from salsa demo.
              mu_ = theta_EB/10;
              muinv = 1/mu_;
              filter_FFT = 1./(abs(H_FFT(b_EB)).^2 + mu_);
              invLS = @(x) real(ifft2(filter_FFT.*fft2( x)));
              invLS = @(xw) callcounter(invLS,xw);
              fprintf('Running SALSA solver...\n')
              % SALSA
              [xMAP, ~, ~, ~, ~, ~, ~] = ...
                  SALSA_v2(y, A1, theta_EB*sigma_EB,...
                  'MU', mu_, ...
                  'AT', AT1, ...
                  'StopCriterion', 1, ...
                  'True_x', x, ...
                  'ToleranceA', tol, ...
                  'MAXITERA', outeriters, ...
                  'Psi', Psi, ...
                  'Phi', op.g, ...
                  'TVINITIALIZATION', 1, ...
                  'TViters', 10, ...
                  'LS', invLS, ...
                  'VERBOSE', 0);
              %Compute MSE with xMAP
              mse_b = 10*log10(norm(x-xMAP,'fro')^2 /dimX);
              results.mse_b=mse_b;
              results.xMAP=xMAP;
              results.snr_true_reconst_image_with_estimate_para = snr_func(x, xMAP);
              [ssimval, ~] = ssim(x, xMAP);
              results.ssim_true_reconst_image_with_estimate_para = ssimval;

            % Sigma^2
              figSigma=figure;
              sig_true = results.sigma^2 * ones(results.last_samp,1);
              plot(results.sigmas(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
              plot(sig_true, 'r--','LineWidth',1.5,'MarkerSize',8);
              xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('$\sigma^2_n$', 'Interpreter', 'latex');title(strcat('$\sigma^2_{EB} = $',num2str(sigma_EB)), 'Interpreter', 'latex');
              legend('$\sigma^2_n$', '$\sigma^2_{true}$', 'Interpreter', 'latex');hold off;


            % Theta
              figTheta=figure;
              plot(results.thetas(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);
              xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\theta_n');title(strcat('$\theta_{EB} = $',num2str(theta_EB)), 'Interpreter', 'latex');

            % b
              true_b = results.b * ones(results.last_samp,1);
              figB=figure;
              plot(results.bs(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
              plot(true_b, 'r--','LineWidth',1.5,'MarkerSize',8);hold off;
              legend('$b_n$', '$b_{true}$', 'Interpreter', 'latex');
              xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('b_n');title(strcat('$b_{EB}$ = ',num2str(b_EB)), 'Interpreter', 'latex');

            % images
              figtrue = figure;
              imagesc(x), colormap gray, axis off, axis equal;
              figmean.Units = 'inches';
              figmean.OuterPosition = [0.25 0.25 10 10];
              title("x")

              figtrue = figure;
              imagesc(y), colormap gray, axis off, axis equal;
              figmean.Units = 'inches';
              figmean.OuterPosition = [0.25 0.25 10 10];
              title("y")

        			figMap = figure;
        			imagesc(xMAP), colormap gray, axis off, axis equal;
        			figMap.Units = 'inches';
        			figMap.OuterPosition = [0.25 0.25 10 10];
              title("x MAP")

		end
  end
end

