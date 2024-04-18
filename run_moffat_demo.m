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
  op.burnIn = (op.samples * 80) / 100;	% iterations we ignore before taking the average over iterates theta_n
  op.sub_sample = 1;
  op.warmup = 15000; % number of warm-up iterations with fixed theta for MYULA sampler
  op.lambdaMax = 2; % max smoothing parameter for MYULA
  op.gammaFrac=0.98; % we set gamma=op.gammaFrac*gamma_max

  % theta
    op.min_th = 1e-3; % projection interval Theta (min theta)
    op.max_th = 1; % projection interval Theta (max theta)

  % Alpha
    op.min_alpha = 1e-2;
    op.max_alpha = 1;

  % Beta
    op.min_beta = 0.1;
    op.max_beta = 10;

  % SNR
    op.BSNR_max = 35;
    op.BSNR_min = 18;

  % True parameters for the psf
    Alpha_v = [ 0.4];
    Beta_v =  [ 3.5];

  % PSF parameters
    psf_size=7;
    op.psf_size = psf_size;

    op.th_init = 0.01; % theta_0 initialisation of the SAPG algorithm
    op.alpha_init = 1; %(op.min_alpha + op.max_alpha) / 2; % alpha_0 initialization of the SAPG algorithm
    op.beta_init = 10; %(-op.min_beta + op.max_beta) / 2;


  % SAPG parameters
    op.d_exp =  0.8;
    op.d_scale =  0.01/op.th_init;

randn('state',1); % Set rnd generators (for repeatability)

%
fix_alpha = 0;
fix_beta = 0;
fix_sigma2 = 0;
op.fix_sigma = fix_sigma2;
op.fix_alpha = fix_alpha;
op.fix_beta = fix_beta;

%% RUN EXPERIMENTS
%----------------------------------------------------------------------
    disp(['Sample = ' num2str(op.samples) 'Warm-up = ' num2str(op.warmup) ...
        'Psf size = ' num2str(psf_size)]);
    %----------------------------------------------------------------------

if 1 == 1
    beta = 3.5;
    alpha = 0.4;
    %%% Fix parameters
    if fix_alpha
        fprintf('\n\n Fix alpha\n\n');
        op.alpha_init = alpha;
    end
    if fix_beta
        fprintf('\n\n Fix beta\n\n');
        op.beta_init = beta;
    end
    %%% True parameters
    op.alpha =alpha;
    op.beta = beta;
    for snr = [30]
        op.BSNR = snr;
        for i_im = [8]
            filename=testImg{i_im};
            [~,fix,~] = fileparts(char(filename));
            name = fix;
            op.name = name;

            %% Generate a noisy and blurred observation vector 'y'
              fprintf(['File:' filename '\n']);
              x = double(imread(strcat('images/', filename)));
              op.x = x;
              dimX = numel(x);
              im_size = size(x);

            % kernel function
              H_FFT  = @(alpha, beta) moffat_psf(im_size, psf_size, alpha, beta);

            %conjugate of the PSF
              HC_FFT = @(alpha, beta) conj(H_FFT(alpha, beta));

            %differential w.r.t alpha
              diff_fftkernel_alpha = @(alpha, beta) diff_moffat_alpha(im_size, psf_size, alpha, beta);

            % differential w.r.t beta
              diff_fftkernel_beta = @(alpha, beta) diff_moffat_beta(im_size, psf_size, alpha, beta);

            %% psf and its gradients
              A = @(x, alpha, beta) real(ifft2(H_FFT(alpha, beta) .* fft2(x)));
              AT = @(x, alpha, beta) real(ifft2(HC_FFT(alpha, beta) .* fft2(x)));
              diff_A_alpha = @(x, alpha, beta) real(ifft2(diff_fftkernel_alpha(alpha, beta) .* fft2(x)));
              diff_A_beta = @(x, alpha, beta) real(ifft2(diff_fftkernel_beta(alpha, beta) .* fft2(x)));

            %% Max eigen value of A through iterative method
              evMax=max_eigenval_Gaussian_Moffat(A,AT,1, 5, im_size, 1e-4, 1e4, 0);

            %% observation y
              Ax =  A(x, alpha, beta);
              sigma =  norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(op.BSNR/10));
              op.sigma = sigma;
              sigma_min= norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(op.BSNR_min/10));
              sigma_max = norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(op.BSNR_max/10));

              op.sigma_init = (sigma_min^2 + sigma_max^2) / 2;
              if op.fix_sigma
                  op.sigma_init = sigma^2;
              end
              op.sigma_min = sigma_min^2;
              op.sigma_max = sigma_max^2;

              y1 = Ax;                                                        %% blur image without noise
              noise = randn(size(Ax));                                        %% noise
              y = y1 + sigma * noise;                                         %% blur image with noise

            %% Experiment setup: functions related to Bayesian model

            % Likelihood and gradients
                op.f = @(x, alpha, beta, sigma2) (norm(y-A(x, alpha, beta),'fro')^2)/(2*sigma2); % p(y|x)∝ exp{-op.f(x)}
                op.gradF = @(x, alpha, beta, sigma2) real(AT(A(x, alpha, beta) - y, alpha, beta)/sigma2);  % Gradient of smooth part f
                op.grad_alpha = @(x, alpha, beta, sigma2) real(sum(sum((diff_A_alpha(x,alpha, beta)) .* (A(x, alpha, beta) - y))) / sigma2);
                op.grad_beta = @(x, alpha, beta, sigma2) real(sum(sum((diff_A_beta(x, alpha, beta)) .* (A(x, alpha, beta) - y))) / sigma2);
                op.gradF_sigma = @(x, alpha, beta, sigma2) ((norm(y - A(x, alpha, beta) ,'fro')^2)/(2*sigma2^2) - dimX/(2*sigma2)) ;
                Lf = @ (sigma2) evMax^2 / sigma2; % define Lipschitz constant of gradient of smooth part
                op.Lf = min(Lf(sigma_min^2), Lf(sigma_max^2));

            % Set algorithm parameters that depend on Lf
                op.lambda =   min(5/op.Lf,op.lambdaMax);%smoothing parameter for MYULA sampler
                op.gamma_max =  1 / (op.Lf+(1/op.lambda));%Lf is the lipschitz constant of the gradient of the smooth part
                op.gamma=   op.gammaFrac * op.gamma_max;%discretisation step MYULA

            % Regulariser % TV norm
                op.g = @(x) TVnorm(x); %g(x) for TV regularizer
                chambolleit = 25;

            % Proximal operator of g(x)
                op.proxG = @(x,lambda,theta) chambolle_prox_TV_stop(x,'lambda',lambda*theta,'maxiter',chambolleit);
                Psi = @(x,th) op.proxG(x, 1, th); % define this format for SALSA solver

            % We use this scalar summary to monitor convergence
                op.logPi = @(x,theta, alpha, beta, sigma) -op.f(x, alpha, beta, sigma) -theta*op.g(x);

            %% Run SAPG Algorithm 1 to compute theta_EB
              tic;
              [theta_EB, alpha_EB, beta_EB, sigma2_EB, results] = SAPG_algorithm_moffat(y, op);
              SAPG_time = toc;
              fprintf('\n\n\t\t=========>>> SAPG Time complexity: %d <<<=========\n\n', SAPG_time);
              results.SAPG_time = SAPG_time;
              results.theta_EB = theta_EB;
              results.alpha_EB = alpha_EB;
              results.beta_EB = beta_EB;
              results.sigma_EB = sigma2_EB;
              results.alpha = alpha;
              results.beta = beta;
              results.sigma = sigma;
              last_alpha = results.last_alpha;
              results.op = op;
              results.x=x;
              results.y=y;
              fprintf('Theta EB: %d\t|\t Alpha EB: %d\t|\t Beta EB: %d\t|\t Last alpha: %d\t|\t Sigma EB: %d', theta_EB, alpha_EB, beta_EB, last_alpha, sigma2_EB);

            %% Solve MYULA problem with theta_EB, alpha_EB, beta_EB and sigma_EB
            % alpha_EB, beta_EB
              tic;
              global calls;
              calls = 0;
              A1 = @(x) A(x, alpha_EB, beta_EB);
              AT1 = @(x) AT(x, alpha_EB, beta_EB);
              A1 = @(x) callcounter(A1,x);
              AT1 = @(x) callcounter(AT1,x);
              inneriters = 1;
              outeriters = 500;
              tol = 1e-5; %taken from salsa demo.
              mu = theta_EB/10;
              muinv = 1/mu;
              filter_FFT = 1./(abs(H_FFT(alpha_EB, beta_EB)).^2 + mu);
              invLS = @(x) real(ifft2(filter_FFT.*fft2( x)));
              invLS = @(xw) callcounter(invLS,xw);
              fprintf('Running SALSA solver...\n')
              % SALSA
              [xMAP, ~, ~, ~, ~, ~, ~] = ...
                  SALSA_v2(y, A1, theta_EB*sigma2_EB,...
                  'MU', mu, ...
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
              mse = 10*log10(norm(x-xMAP,'fro')^2 /dimX);
              alpha_mse_time = toc;
              [ssimval, ~] = ssim(x, xMAP);

             % approximated psf
               app_psf = psf_moffat(op.psf_size, alpha_EB, beta_EB);
               true_psf = psf_moffat(op.psf_size, alpha, beta);

            % Sigma^2
              figSigma=figure;
              sig_true = results.sigma^2 * ones(results.last_samp,1);
              plot(results.sigmas(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
              plot(sig_true, 'r--','LineWidth',1.5,'MarkerSize',8);
              xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('$\sigma^2_n$', 'Interpreter', 'latex');title(strcat('$\sigma^2_{EB}$ = ',num2str(sigma2_EB)), 'Interpreter', 'latex');
              legend('$\sigma^2_n$', '$\sigma^2_{true}$', 'Interpreter', 'latex');hold off;

            % Theta
              figTheta=figure;
              plot(results.thetas(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);
              xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\theta_n');title(strcat('$\theta_{EB}$ = ',num2str(theta_EB)), 'Interpreter', 'latex');

            % alpha
              true_alpha = results.alpha * ones(results.last_samp,1);
              figAlpha=figure;
              plot(results.alphas(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
              plot(true_alpha, 'r--','LineWidth',1.5,'MarkerSize',8);hold off;
              legend('$\alpha_n$', '$\alpha_{true}$', 'Interpreter', 'latex');
              xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\alpha_n');title(strcat('$\alpha_{EB}$ = ',num2str(alpha_EB)), 'Interpreter', 'latex');

            % beta
              true_beta = results.beta * ones(results.last_samp,1);
              figBeta=figure;
              plot(results.betas(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
              plot(true_beta, 'r--','LineWidth',1.5,'MarkerSize',8);hold off;
              legend('$\beta_n$', '$\beta_{true}$', 'Interpreter', 'latex');
              xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\beta_n');title(strcat('$\beta_{EB}$ = ',num2str(beta_EB)), 'Interpreter', 'latex');

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
