
%% Experiment 4.2.1 - Deblurring with Total-Variation prior
%{
  Experiment 4.2.1 of the SIAM article [1]:
  Deblurring with Total-Variation prior

  Computes the maximum marignal likelihood estimation of the
  regularisation parameter using Algorithm 1 from [1], i.e.
  the algorithm for homogeneous regularisers and scalar theta.
  
  It runs Algorithm 1 for 10 test images, and computes the MAP estimator
  using the estimated value theta_EB and the SALSA solver.

  All results are saved in a <img_name>_results.mat file in a 'results'
  directory. This .mat file contains a structure with:
  
  -execTimeFindTheta: time it took to compute theta_EB in seconds
  -last_samp: number of iteration where the SAPG algorithm was stopped
  -logPiTrace_WU: an array with k*log pi(x_n|y,theta_0) during warm-up
   (k is some unknown proportionality constant)
  -logPiTraceX: an array with k*log pi(x_n|y,theta_n) for the SAPG
  -gXTrace: saves the regulariser evolution through iterations g(x_n)
  -mean_theta: theta_EB computed taking the average over all iterations
   skipping the appropriate burn-in iterations
  -last_theta: simply the last value of the iterate theta_n computed
  -thetas: full array of theta_n for all iterations
  -options: structure with all the options used to run the algorithm
  -mse: mean squarred error after computing x_MAP with theta_EB
  -xMAP: estimated x_MAP
  -x: original image
  -y: observations (blurred and noisy image)

  [1] A. F. Vidal, V. De Bortoli, M. Pereyra, and D. Alain, Maximum
  likelihood estimation of regularisation parameters in high-dimensional
  inverse problems: an empirical bayesian approach. Part I:Methodology and 
  Experiments.
  
  Prerequired libraries:  
  - SALSA: http://cascais.lx.it.pt/~mafonso/salsa.html

%}
%  ===================================================================
%% Create array with test images names
clear all;clc;
testImg=dir('images');
testImg={testImg(3:end).name};
%Check that test images are accesible
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
op.samples =1600; % max iterations for SAPG algorithm to estimate theta
op.stopTol=1e-3; % tolerance in relative change of theta_EB to stop the algorithm
op.burnIn=20;	% iterations we ignore before taking the average over iterates theta_n

tau = .4;
op.th_init = 0.01 ; % theta_0 initialisation of the SAPG algorithm
op.min_th=1e-3; % projection interval Theta (min theta)
op.max_th=1; % projection interval Theta (max theta)
op.min_to = 0.3;
op.max_to = 1;
op.to_init = .9;
% delta(i) for SAPG algorithm defined as: op.d_scale*( (i^(-op.d_exp)) / numel(x) );
op.d_exp =  0.8;
op.d_scale =  0.1/op.th_init;

%MYULA parameters
op.warmup = 300 ;% number of warm-up iterations with fixed theta for MYULA sampler
op.lambdaMax = 2; % max smoothing parameter for MYULA
lambdaFun= @(Lf) min((5/Lf),op.lambdaMax);
gamma_max = @(Lf,lambda) 1/(Lf+(1/lambda));%Lf is the lipschitz constant of the gradient of the smooth part
op.gammaFrac=0.98; % we set gamma=op.gammaFrac*gamma_max

randn('state',1); % Set rnd generators (for repeatability)
%% RUN EXPERIMENTS
for i_im = 1:length(testImg)
    for bsnr=[40]
        fprintf('SNR=%d\n',bsnr); op.BSNRdb=bsnr;
        filename=testImg{i_im};
        %% Generate a noisy and blurred observation vector 'y'
        %%%%%  original image 'x'
        fprintf(['File:' filename '\n']);
        x = double(imread(filename));
        dimX = numel(x);
        
        %%%%%  blur operator
        blur_length=31;
        [X, Y] = meshgrid(1:blur_length);
        
        % kernel function
        H_FFT  = @(to) fftkernel_f(x, X, Y, blur_length, to);
        HC_FFT = @(to) conj(H_FFT(to));
        dif_fftkernel = @(to) dif_fftkernel_f(x, X, Y, blur_length, to);
        
        % Gaussian kernel
        A = @(x, to) real(ifft2(H_FFT(to) .* fft2(x)));
        AT = @(x, to) real(ifft2(HC_FFT(to) .* fft2(x)));
        dif_A = @(x, to) real(ifft2(dif_fftkernel(to) .* fft2(x)));
        
        % Max eigen value of A through iterative method
        evMax=max_eigenval(A,AT,tau, size(x), 1e-4, 1e4, 0); 
        fprintf('evmax=%d\n',evMax);
        
        %%%%% observation y
        Ax = A(x, tau);  
        sigma = norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(op.BSNRdb/10));
        sigma2 = sigma^2; op.sigma=sigma;
        y = Ax + sigma*randn(size(Ax));
        op.y = y;
        
        %% Experiment setup: functions related to Bayesian model
        %%%% Regulariser
        % TV norm
        op.g = @(x) TVnorm(x); %g(x) for TV regularizer
        chambolleit = 25;
        
        % Proximal operator of g(x)         
        op.proxG = @(x,lambda,theta) chambolle_prox_TV_stop(x,'lambda',lambda*theta,'maxiter',chambolleit);
        Psi = @(x,th) op.proxG(x,1,th); % define this format for SALSA solver
                         
        %%%% Likelihood (data fidelity)
        op.f = @(x, tau) (norm(y-A(x, tau),'fro')^2)/(2*sigma2); % p(y|x)∝ exp{-op.f(x)}
        op.gradF = @(x, tau) real(AT(A(x, tau)-y, tau)/sigma2);  % Gradient of smooth part f
        op.grad_t = @(x, tau) sum(sum((dif_A(x, tau)) .* (A(x, tau) - y)));
        Lf = (evMax/sigma)^2; % define Lipschitz constant of gradient of smooth part

        % We use this scalar summary to monitor convergence
        op.logPi = @(x,theta, tau) -op.f(x, tau) -theta*op.g(x);
        
        %%%% Set algorithm parameters that depend on Lf
        op.lambda=lambdaFun(Lf);%smoothing parameter for MYULA sampler
        op.gamma=op.gammaFrac*gamma_max(Lf,op.lambda);%discretisation step MYULA
        
        %% Run SAPG Algorithm 1 to compute theta_EB
        [theta_op,tau_op, results]=SAPG_algorithm_1(y,op);
        op.theta_op = theta_op;
        op.tau_op = tau_op;
        
        %% Solve MYULA problem with theta_op and tau_op       
        x_op = myula(op, x);
       
        results.x_op=x_op;
        results.x=x;
        results.y=y;
        
        %% Save Results
        %Create directories
        %subdirID: creates aditional folder with this name to store results
        subdirID = ['algo1_tol' char(num2str(op.stopTol)) '_maxIter' char(num2str(op.samples)) ];
        dirname = 'results/4.2.1-deblur-TV';
        subdirname=['BSNR' char(num2str(op.BSNRdb)) '/' subdirID ];
        if ~exist(['./' dirname],'dir')
            mkdir('results');
        end
        if ~exist(['./' dirname '/' subdirname ],'dir')
            mkdir(['./' dirname '/' subdirname ]);
        end
        
        % Save data
        [~,name,~] = fileparts(char(filename));
        save(['./' dirname '/' subdirname '/' name '_results.mat'], '-struct','results');
        close all;
        
        % Save figures (degraded and recovered)
        f1=figure; imagesc(results.x), colormap gray, axis equal, axis off;
        f2=figure; imagesc(results.y),  colormap gray, axis equal, axis off;
        f3=figure; imagesc(results.x_op), colormap gray, axis equal, axis off;
        
        
    end
end
