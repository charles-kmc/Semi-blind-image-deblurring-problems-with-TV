% Charlesquin Kemajou Mbakam: mbakamkemajou@gmail.org
% Heriot-Watt University
% date: Juin 2022

%% Experiment: Deblurring with Total-Variation prior: Guassian Kernel with unknown bandwidth.

addpath([pwd,'/utils']);
addpath([pwd,'/SALSA']);
addpath([pwd,'/SAPG']);

clear all;clc;

%% Create array with test images names

testImg = dir('images');
% testImg = dir('image');
testImg = {testImg(3:end).name};

%Check that tesexitt images are accesible
if sum(size(testImg)) == 0
    error('No images found, please check that the images directory is in the MATLAB path');
end

%Check that SALSA solver is in the MATLAB path
if exist('SALSA_v2')==0
    error('SALSA package not found. Please make sure that the salsa solver is added to your MATLAB path.  The code can be obtained from http://cascais.lx.it.pt/~mafonso/salsa.html');
end


Dir = 'Results_Gaussian/March/25_23';

%% Hyper-parameters
%--------------------------------------------------------------------------
c.sigma = 1000;
c.theta = 0.01;
c.w1  = 10;
c.w2  = 10;
c.lam = 1;
c.gam = 1;

fix_sigma = 0;
fix_w1 = 1;
fix_w2 = 1;
%--------------------------------------------------------------------------

%% MYULA Parameters Setup
op.samples = 20000;
op.stopTol = 1e-5;
op.burnIn = (op.samples * 80) / 100;
op.warmup = 15000;
op.lambdaMax = 2;
op.gammaFrac=0.98;

%% SAPG Parameters Setup
op.min_th = 1e-3;
op.max_th = 1;
op.min_w1 = 0.1;
op.max_w1 = 1;
op.min_w2 = 0.1;
op.max_w2 = 1;

% signal noise ratio
op.BSNR_max = 45;
op.BSNR_min = 15;

% Initial value for parameters
op.th_init = 0.01;
op.w1_init = 0.5;
op.w2_init = 0.3;

op.d_exp =  0.8;
op.d_scale =  0.01/op.th_init;


psf_size= 7;
phi = 0;
op.phi = phi;
w1 = [0.4];
w2 = [0.3];
op.w1 = w1;
op.w2 = w2;

op.fix_w1 = fix_w1;
op.fix_w2 = fix_w2;
op.fix_sigma = fix_sigma;

all = 0;
randn('state',1);


%% RUN EXPERIMENTS
%----------------------------------------------------------------------
    disp(['Sample = ' num2str(op.samples) ' Warm-up = ' num2str(op.warmup) ...
        ' Psf size = ' num2str(phi)]);
    %----------------------------------------------------------------------

if 1 == 1
    for snr = [30]
        op.BSNR = snr;
		for i_im = [8]
			dirname = Dir;
			if fix_w1
				op.w1_init = w1;
			end
			if fix_w2
				op.w2_init = w2;
			end

			filename=testImg{i_im};
			[~,fix,~] = fileparts(char(filename));
			name = fix;
			op.name = name;

			%% Generate a noisy and blurred observation vector 'y'

			fprintf(['File:' filename '\n']);
			x = double(imread(strcat('images/', filename)));
			dimX = numel(x);
			op.x = x;


			%%  blur operator
			op.psf_size = psf_size;
			im_size = size(x);

			h  = @(a,b) Gaussian_psf(psf_size, a,b, phi);
			h_adj = @(a,b) h(a,b)';
			H_FFT = @(a,b) resize(h(a,b),im_size);
			HC_FFT = @(a,b) conj(H_FFT(a,b));

			% Gradient functions
			dif_fftkernel_w1 = @(a,b) diff_fftgaus_w1(im_size, psf_size, a,b, phi);
			dif_fftkernel_w2 = @(a,b) diff_fftgaus_w2(im_size, psf_size, a,b, phi);

			%% Gaussian operators
			A  = @(x, a,b) real(ifft2(H_FFT(a,b) .* fft2(x)));
			AT = @(x, a,b) real(ifft2(HC_FFT(a,b) .* fft2(x)));
			dif_w1 = @(x, a,b) real(ifft2(dif_fftkernel_w1(a,b) .* fft2(x)));
			dif_w2 = @(x, a,b) real(ifft2(dif_fftkernel_w2(a,b) .* fft2(x)));

			%% Max eigen value of A through iterative method
			evMax = max_eigenval_Gaussian_Moffat(A,AT,1,1, im_size, 1e-4, 1e4, 0);

			%% observation y
			Ax =  A(x, w1,w2);

			% computing sigma^2 and its initialisation
			sigma = norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(op.BSNR/10));

			op.sigma = sigma;
			sigma_min= norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(op.BSNR_min/10));
			sigma_max = norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(op.BSNR_max/10));
			s_min = min(sigma_min^2, sigma_max^2);
			s_max = max(sigma_min^2, sigma_max^2);

			if fix_sigma
				disp('**** Fix sigma^2****');
				op.sigma_init = sigma^2;
			else
				op.sigma_init = (sigma_min^2 + sigma_max^2) / 2;
			end
			op.sigma_min = sigma_min^2;
			op.sigma_max = sigma_max^2;

			%% Model
			y1 = Ax;
			noise = randn(size(Ax));
			y = y1 + sigma * noise;

			 %% Likelihood and Gradients
			op.f = @(x, a,b, sigma2) (norm(y-A(x, a,b),'fro')^2)/(2*sigma2); % p(y|x)∝ exp{-op.f(x)}
			op.gradF = @(x, a,b, sigma2) real(AT(A(x, a,b) - y, a,b)/sigma2);  % Gradient of smooth part f
			op.grad_w1 = @(x, a,b, sigma2) real(sum(sum((dif_w1(x, a,b)) .* (A(x, a,b) - y))) / sigma2);
			op.grad_w2 = @(x, a,b, sigma2) real(sum(sum((dif_w2(x, a,b)) .* (A(x, a,b) - y))) / sigma2);
			op.gradF_sigma = @(x, a,b, sigma2) ((norm(y - A(x, a,b) ,'fro')^2)/(2*sigma2^2) - dimX/(2*sigma2)) ;

			% Lipschitz constant
			lf = @ (sigma2) evMax^2 / sigma2;
			op.Lf = min(lf(sigma_min^2),lf(sigma_max^2));

			%% Setting MYULA step size
			op.lambda= min((5/op.Lf),op.lambdaMax);
			op.gamma_max =  1 / (op.Lf+(1/op.lambda));
			op.gamma= op.gammaFrac * op.gamma_max;

			%% Regulariser % TV norm
			op.g = @(x) TVnorm(x); %g(x) for TV regularizer
			chambolleit = 25;

			%% Proximal operator of g(x)
			op.proxG = @(x,theta) chambolle_prox_TV_stop(x,'lambda',op.lambda*theta,'maxiter',chambolleit);
			Psi = @(x,th) chambolle_prox_TV_stop(x,'lambda',th,'maxiter',chambolleit);

			% We use this scalar summary to monitor convergence
			op.logPi = @(x,theta, a, b, sigma) -op.f(x, a, b, sigma) -theta*op.g(x);

			%% Run SAPG Algorithm 1 to compute theta_EB, w1_EB, w2_EB and sigma2_EB
			tic;
			[theta_EB, w1_EB, w2_EB, sigma_EB, results] = SAPG_algorithm_Guassian(y, op, c);
			SAPG_time = toc;
			fprintf('\n\n\t\t=========>>> SAPG Time complexity: %d <<<=========\n\n', SAPG_time);
			op.theta_EB = theta_EB;
			op.w1_EB = w1_EB;
			op.w2_EB = w2_EB;
			results.x=x;
			results.y=y;
			fprintf('Theta : %d\t|\t w1: %d\t|\t w2: %d\t|\t Sigma: %d', theta_EB, w1_EB, w2_EB, sigma_EB);

			%% Solve MYULA problem with theta_op and tau_op
			global calls;
			calls = 0;

			%% w1_EB, w2_EB, theta_EB, sigma2_EB
			tic;
			A1 = @(x) A(x, w1_EB, w2_EB);
			AT1 = @(x) AT(x, w1_EB, w2_EB);
			A1 = @(x) callcounter(A1,x);
			AT1 = @(x) callcounter(AT1,x);
			inneriters = 1;
			outeriters = 500;
			tol = 1e-5; %taken from salsa demo.
			mu = theta_EB/10;
			muinv = 1/mu;
			filter_FFT = 1./(abs(H_FFT(w1_EB, w2_EB)).^2 + mu);
			invLS = @(x) real(ifft2(filter_FFT.*fft2( x )));
			invLS = @(xw) callcounter(invLS,xw);
			fprintf('Running SALSA solver...\n')
			% SALSA
			[xMAP, ~, ~, ~, ~, ~, ~] = ...
				SALSA_v2(y, A1, theta_EB*sigma_EB,...
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
			[ssimval, ~] = ssim(x, xMAP);

			% Sigma^2
			figSigma=figure;
			sig = sigma^2 * ones(results.last_samp,1);
			plot(results.sigmas(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
			plot(sig, 'r');hold off;
			xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('$\sigma^2_n$', 'Interpreter', 'latex');
			title(strcat('$\sigma^2_{op}$ = ', num2str(sigma_EB)), 'Interpreter', 'latex');
			legend('$\sigma^2_n$', '$\sigma^2_{true}$', 'Interpreter', 'latex');hold off;


			% Theta
			figTheta=figure;
			plot(results.thetas(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);
			xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\theta_n');title(strcat('\theta_{op} = ', num2str(theta_EB)))

			% w1
			true_w1 = w1 * ones(results.last_samp,1);
			figw1=figure;
			plot(results.w1s(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
			plot(true_w1, 'r');hold off;
			legend('$w1_n$', '$w1_{true}$', 'Interpreter', 'latex');hold off;
			xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('w1_n');title(strcat('w1_{op} = ', num2str(w1_EB)))

			% w2
			true_w2 = w2 * ones(results.last_samp,1);
			figw2=figure;
			plot(results.w2s(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
			plot(true_w2, 'r');hold off;
			legend('$w2_n$', '$w2_{true}$', 'Interpreter', 'latex');hold off;
			xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('w2_n');title(strcat('w2_{op} = ', num2str(w2_EB)))


			ffigtrue = figure;
		  	imagesc(x), colormap gray, axis off, axis equal;
		  	figmean.Units = 'inches';
		  	figmean.OuterPosition = [0.25 0.25 10 10];
		  	title("x")

             figtrue = figure;
             imagesc(y), colormap gray, axis off, axis equal;
             figmean.Units = 'inches';
             figmean.OuterPosition = [0.25 0.25 10 10];
             title("y")

            %  figmean = figure;
			%  imagesc(results.posteriormean), colormap gray, axis off, axis equal;
			%  figmean.Units = 'inches';
			%  figmean.OuterPosition = [0.25 0.25 10 10];
            %  title("posterior mean")

			 figMap = figure;
			 imagesc(xMAP), colormap gray, axis off, axis equal;
			 figMap.Units = 'inches';
			 figMap.OuterPosition = [0.25 0.25 10 10];
             title("x MAP")

		end
    end
end
