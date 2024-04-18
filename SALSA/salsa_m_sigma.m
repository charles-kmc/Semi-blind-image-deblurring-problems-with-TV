
%% Experiment 4.2.1 - Deblurring with Total-Variation prior: Guassian Kernel with unknown bandwidth.

%% Create array with test images names
clear all;clc;
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
op.samples = 5000; % max iterations for SAPG algorithm to estimate theta
op.stopTol = 1e-5; % tolerance in relative change of theta_EB to stop the algorithm
op.tol2 = 1e-2;  % tolerance for tau
op.burnIn =1000;	% iterations we ignore before taking the average over iterates theta_n

op.min_th = 1e-3; % projection interval Theta (min theta)
op.max_th = 1; % projection interval Theta (max theta)
op.min_to = 0.1;
op.max_to = 1;

op.BSNR = 30;
op.BSNR_max = 40;
op.BSNR_min = 25;

tau = 0.4;

op.th_init = 0.01; % theta_0 initialisation of the SAPG algorithm
op.to_init = 0.6; % tau_0 initialization of the SAPG algorithm

% delta(i) for SAPG algorithm defined as: op.d_scale*( (i^(-op.d_exp)) / numel(x) );
op.d_exp =  0.8;
op.d_scale =  0.01/op.th_init;

%MYULA parameters
op.warmup = 3000 ; % number of warm-up iterations with fixed theta for MYULA sampler
op.lambdaMax = 2; % max smoothing parameter for MYULA
op.gammaFrac=0.98; % we set gamma=op.gammaFrac*gamma_max
randn('state',1); % Set rnd generators (for repeatability)
%% MSE
vec_sigma = 0.1:1:10.1;
Theta_op = 5e-3:0.01:6e-2;%[1.7e-3 1e-2 1.2e-2 1.5e-2 1.8e-2 2e-2 2.5e-2 3e-2 4e-2 5e-2 6e-2];
Tau_op = 0.1:0.05:0.7;%[0.3 0.4 0.5 0.6 0.7 0.8];
MSE_theta(length(Theta_op)) = 0;
MSE_tau(length(Tau_op)) = 0;
%% RUN EXPERIMENTS
for i_im = 10:length(testImg)
    filename=testImg{i_im};
    [~,fix,~] = fileparts(char(filename));
    %% Generate a noisy and blurred observation vector 'y'
    %%%%%  original image 'x'
    fprintf(['File:' filename '\n']);
    x = double(imread(strcat('images/', filename)));
    dimX = numel(x);
    
    %%  blur operator
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
    
    %% Max eigen value of A through iterative method
    evMax=max_eigenval(A,AT,tau, size(x), 1e-4, 1e4, 0);
    fprintf('evmax=%d\n',evMax);
    %% observation y
    Ax =  A(x, tau);
    sigma = norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(op.BSNR/10));
    fprintf('Sigma = %d\n', sigma);
    op.sigma = sigma;
    sigma_min= norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(op.BSNR_min/10));
    sigma_max = norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(op.BSNR_max/10));
    fprintf('Sigma max = %d\t\t sigma min = %d\n',  sigma_min^2, sigma_max^2);
    
    op.sigma_init = (sigma_min^2 + sigma_max^2) / 2;
    op.sigma_min = sigma_min^2;
    op.sigma_max = sigma_max^2;
    y = Ax + sigma * randn(size(Ax));
    
    %% test
    if 1==2
        v(1) = 0;
        i = 1;
        for a= 25:40
            v(i) = (norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(a/10)))^2;
            i = i+1;
        end
        plot(v,'*');
        sdf
    end
    
    %% Experiment setup: functions related to Bayesian model
    
    %% Likelihood (data fidelity)
    op.f = @(x, tau, sigma2) (norm(y-A(x, tau),'fro')^2)/(2*sigma2); % p(y|x)∝ exp{-op.f(x)}
    op.gradF = @(x, tau, sigma2) real(AT(A(x, tau) - y, tau)/sigma2);  % Gradient of smooth part f
    op.grad_t = @(x, tau, sigma2) real(sum(sum((dif_A(x, tau)) .* (A(x, tau) - y))) / sigma2);
    op.gradF_sigma = @(x, tau, sigma2) ((norm(y - A(x, tau) ,'fro')^2)/(2*sigma2^2) - dimX/(2*sigma2)) ;
    op.Lf = @ (sigma2) evMax^2 / sigma2; % define Lipschitz constant of gradient of smooth part
    
    
    
    
    %% Set algorithm parameters that depend on Lf
    %op.lambdaFun= @(sigma) min((5/op.Lf(sigma)),op.lambdaMax);
    op.lambda=  @ (sigma2) min((5/op.Lf(sigma2)),op.lambdaMax);%smoothing parameter for MYULA sampler
    op.gamma_max = @(sigma2) 1 / (op.Lf(sigma2)+(1/op.lambda(sigma2)));%Lf is the lipschitz constant of the gradient of the smooth part
    op.gamma= @ (sigma2) op.gammaFrac * op.gamma_max(sigma2);%discretisation step MYULA
    
    % Regulariser % TV norm
    op.g = @(x) TVnorm(x); %g(x) for TV regularizer
    chambolleit = 25;
    
    % Proximal operator of g(x)
    op.proxG = @(x,sigma,theta) chambolle_prox_TV_stop(x,'lambda',op.lambda(sigma)*theta,'maxiter',chambolleit);
    Psi = @(x,th) chambolle_prox_TV_stop(x,'lambda',th,'maxiter',chambolleit); % op.proxG(x, sigma, th); % define this format for SALSA solver
    
    % We use this scalar summary to monitor convergence
    op.logPi = @(x,theta, tau, sigma) -op.f(x, tau, sigma) -theta*op.g(x);
    
    %% Run SAPG Algorithm 1 to compute theta_EB
    tic;
    [theta_EB, tau_EB, sigma_EB, results] = SAPG_algorithm_sigma(y, op);
    SAPG_time = toc;
    results.SAPG_time = SAPG_time;
    op.theta_EB = theta_EB;
    op.tau_EB = tau_EB;
    last_tau = results.last_tau;
    results.x=x;
    results.y=y;
    fprintf('Theta optimal: %d\t|\t Tau optimal: %d\t|\t Last tau: %d\t|\t Sigma: %d', theta_EB, tau_EB, last_tau, sigma_EB);
    
    %% Solve MYULA problem with theta_op and tau_op
    global calls;
    calls = 0;
    
    %% tau_EB
    tic;
    A1 = @(x) A(x, tau_EB);
    AT1 = @(x) AT(x, tau_EB);
    A1 = @(x) callcounter(A1,x);
    AT1 = @(x) callcounter(AT1,x);
    inneriters = 1;
    outeriters = 2000;%500 is used in salsa demo.
    tol = 1e-5; %taken from salsa demo.
    mu = theta_EB/10;
    muinv = 1/mu;
    filter_FFT = 1./(abs(H_FFT(tau_EB)).^2 + mu);
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
    tau_mse_time = toc;
    results.tau_mse_time = tau_mse_time;
    results.mse=mse;
    results.xMAP=xMAP;
    
    
    %% sigma^2 MSE
    mse_sigma(1) = 0;
    mse_sigma(length(vec_sigma)) = 0;
    tic;
    for jj =1:length(vec_sigma)
        sig2 = vec_sigma(jj);
        A1 = @(x) A(x, tau_EB);
        AT1 = @(x) AT(x, tau_EB);
        A1 = @(x) callcounter(A1,x);
        AT1 = @(x) callcounter(AT1,x);
        inneriters = 1;
        outeriters = 2000;%500 is used in salsa demo.
        tol = 1e-5; %taken from salsa demo.
        mu = theta_EB/10;
        muinv = 1/mu;
        filter_FFT = 1./(abs(H_FFT(tau_EB)).^2 + mu);
        invLS = @(x) real(ifft2(filter_FFT.*fft2( x )));
        invLS = @(xw) callcounter(invLS,xw);
        fprintf('Running SALSA solver...\n')
        % SALSA
        [xMAP, ~, ~, ~, ~, ~, ~] = ...
            SALSA_v2(y, A1, theta_EB*sig2,...
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
        msejj = 10*log10(norm(x-xMAP,'fro')^2 /dimX);
        mse_sigma(jj) = msejj;
    end
    sigma_mse_time = toc;
    results.sigma_mse_time = sigma_mse_time;
    results.mse_sigma=mse_sigma;
    [val, ind1] = min(mse_sigma(:));
    sigma_oracle = vec_sigma(ind1);
    mse_oracle = val;
    
    
    %% last tau
    tic;
    A1 = @(x) A(x, last_tau);
    AT1 = @(x) AT(x, last_tau);
    A1 = @(x) callcounter(A1,x);
    AT1 = @(x) callcounter(AT1,x);
    inneriters = 1;
    outeriters = 2000;%500 is used in salsa demo.
    tol = 1e-5; %taken from salsa demo.
    mu = theta_EB/10;
    muinv = 1/mu;
    filter_FFT = 1./(abs(H_FFT(last_tau)).^2 + mu);
    invLS = @(x) real(ifft2(filter_FFT.*fft2( x )));
    invLS = @(xw) callcounter(invLS,xw);
    fprintf('Running SALSA solver...\n')
    % SALSA
    [xMAP_last, ~, ~, ~, ~, ~, ~] = ...
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
    mse_last = 10*log10(norm(x-xMAP_last,'fro')^2 /dimX);
    tau_N_mse_time = toc;
    results.tau_N_mse_time = tau_N_mse_time;
    results.mse_last=mse_last;
    results.xMAP_last=xMAP_last;
    
    %% tau_true
    tic;
    A_ = @(x) A(x, tau);
    AT_ = @(x) AT(x, tau);
    A_ = @(x) callcounter(A_,x);
    AT_ = @(x) callcounter(AT_,x);
    inneriters = 1;
    outeriters = 2000;%500 is used in salsa demo.
    tol = 1e-5; %taken from salsa demo.
    mu = theta_EB/10;
    muinv = 1/mu;
    filter_FFT = 1./(abs(H_FFT(tau)).^2 + mu);
    invLS = @(x) real(ifft2(filter_FFT.*fft2( x )));
    invLS = @(xw) callcounter(invLS,xw);
    fprintf('Running SALSA solver...\n')
    % SALSA
    [xMAP, ~, ~, ~, ~, ~, ~] = ...
        SALSA_v2(y, A_, theta_EB*sigma_EB,...
        'MU', mu, ...
        'AT', AT_, ...
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
    mse_true_tau = 10*log10(norm(x-xMAP,'fro')^2 /dimX);
    true_tau_mse_time = toc;
    results.true_tau_mse_time = true_tau_mse_time;
    results.mse_true_tau = mse_true_tau;
    
    %% fix theta
    tic;
    for jj = 1:length(Tau_op)
        to = Tau_op(jj);
        theta = theta_EB;
        %% Solve MYULA problem with theta_op and tau_op
        global calls;
        calls = 0;
        %A = @(x) callcounter(A,x);
        %AT = @(x) callcounter(AT,x);
        A2 = @(x) A(x, to);
        AT2 = @(x) AT(x, to);
        
        A2 = @(x) callcounter(A2,x);
        AT2 = @(x) callcounter(AT2,x);
        inneriters = 1;
        outeriters = 2000;%500 is used in salsa demo.
        tol = 1e-5; %taken from salsa demo.
        mu = theta/10;
        muinv = 1/mu;
        filter_FFT = 1./(abs(H_FFT(to)).^2 + mu);
        invLS = @(x) real(ifft2(filter_FFT.*fft2( x )));
        invLS = @(xw) callcounter(invLS,xw);
        fprintf('Running SALSA solver...\n')
        % SALSA
        [xMAP, ~, ~, ~, ~, ~, ~] = ...
            SALSA_v2(y, A2, theta*sigma_EB,...
            'MU', mu, ...
            'AT', AT2, ...
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
        MSE_tau( jj) = 10*log10(norm(x-xMAP,'fro')^2 /dimX);
        fprintf('MSE:%d\t theta: %d\t tau:%d\n', MSE_tau(jj), theta, to);
    end
    fix_theta_mse_time = toc;
    results.fix_theta_mse_time= fix_theta_mse_time;
    
    results.MSE_tau=MSE_tau;
    
    %% fix tau
    tic;
    for jj = 1:length(Theta_op)
        to = tau_EB;%last_tau;
        theta = Theta_op(jj);
        %% Solve MYULA problem with theta_op and tau_op
        global calls;
        calls = 0;
        %A = @(x) callcounter(A,x);
        %AT = @(x) callcounter(AT,x);
        A2 = @(x) A(x, to);
        AT2 = @(x) AT(x, to);
        
        A2 = @(x) callcounter(A2,x);
        AT2 = @(x) callcounter(AT2,x);
        inneriters = 1;
        outeriters = 2000;%500 is used in salsa demo.
        tol = 1e-5; %taken from salsa demo.
        mu = theta/10;
        muinv = 1/mu;
        filter_FFT = 1./(abs(H_FFT(to)).^2 + mu);
        invLS = @(x) real(ifft2(filter_FFT.*fft2( x )));
        invLS = @(xw) callcounter(invLS,xw);
        fprintf('Running SALSA solver...\n')
        % SALSA
        [xMAP, ~, ~, ~, ~, ~, ~] = ...
            SALSA_v2(y, A2, theta*sigma_EB,...
            'MU', mu, ...
            'AT', AT2, ...
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
        MSE_theta( jj) = 10*log10(norm(x-xMAP,'fro')^2 /dimX);
        fprintf('MSE:%d\t theta: %d\t tau:%d\n', MSE_theta(jj), theta, to);
    end
    fix_tau_mse_time = toc;
    results.fix_tau_mse_time = fix_tau_mse_time;
    
    results.MSE_theta=MSE_theta;
    [m1, n1] = min(MSE_theta(:));
    x_oracle = Theta_op(n1);
    y_oracle = m1;
    
    %% Save Results
    %Create directories
    %subdirID: creates aditional folder with this name to store results
    subdirID = ['algo1_tol' char(num2str(op.stopTol)) '_maxIter' char(num2str(op.samples)) ];
    dirname = 'result_sigma18/4.2.1-deblur-TV';
    subdirname=['BSNR/' subdirID ];
    subdirname  = strcat(fix, subdirname);
    if ~exist(['./' dirname ],'dir')
        mkdir('result_sigma18');
    end
    if ~exist(['./' dirname '/' subdirname ],'dir')
        mkdir(['./' dirname '/' subdirname ]);
    end
    
    % Save data
    [~,name,~] = fileparts(char(filename));
    
    save(['./' dirname '/' subdirname '/' name '_results.mat'], '-struct','results');
    
    %% tau
    figMSE_tau=figure;
    plot(Tau_op, results.MSE_tau, 'b');hold on;
    plot( tau_EB,results.mse, '+');
    plot( tau,results.mse_true_tau, '*');
    plot( last_tau,results.mse_last, 'o');
    title('MSE \alpha');xlabel('\alpha');
    legend('MSE','$\bar{\alpha_n}$',  '$\alpha_{oracle}$', '$\alpha_{N}$', 'Interpreter', 'latex');hold off;
    saveas(figMSE_tau,['./' dirname '/' subdirname '/' name '_MSE_alpha.png']);
    
    %% theta
    figMSE_theta=figure;
    plot(Theta_op, results.MSE_theta, 'b');hold on;
    plot( theta_EB, results.mse, 'o');
    plot( x_oracle, y_oracle, '*');
    title('MSE \theta');xlabel('\theta');
    legend('MSE','$\bar{\theta_n}$', '$\theta_{oracle}$', 'Interpreter', 'latex');hold off;
    saveas(figMSE_theta,['./' dirname '/' subdirname '/' name '_MSE_theta.png']);
    
    %% mse sigma^2
    figMSE_sigma=figure;
    plot(vec_sigma, results.mse_sigma, 'b');hold on;
    plot( sigma_EB, results.mse, 'o');
    plot( sigma_oracle, mse_oracle, '*');
    title('MSE \sigma^2');xlabel('\sigma^2');
    legend('MSE','$\bar{\sigma^2_n}$', '$\sigma^2_{oracle}$', 'Interpreter', 'latex');hold off;
    saveas(figMSE_sigma,['./' dirname '/' subdirname '/' name '_MSE_sigma.png']);
    
    %% Save figures (degraded and recovered)
    f1=figure; imagesc(results.x), colormap gray, axis equal, axis off;
    f2=figure; imagesc(results.y),  colormap gray, axis equal, axis off;
    f3=figure; imagesc(results.xMAP), colormap gray, axis equal, axis off;
    f4=figure; imagesc(results.xMAP_last), colormap gray, axis equal, axis off;
    
    saveas(f2,['./' dirname '/' subdirname '/' name '_degraded.png']);
    saveas(f3,['./' dirname '/' subdirname '/' name '_estim_mean_tau.png']);
    saveas(f4,['./' dirname '/' subdirname '/' name '_estim_last_tau.png']);
    saveas(f1,['./' dirname '/' subdirname '/' name '_orig.png']);
    close all;
    if save_logpi_plots
        if(op.warmup>0)
            % Save images of logpitrace
            figLogPiTraceWarmUp=figure;
            plot(results.logPiTrace_WU(2:end),'b','LineWidth',1.5,'MarkerSize',8);   hold on;
            title('\propto log p(X^n|y,\theta, \alpha) during warm-up');xlabel('Iteration (n)');grid on; hold off;
            saveas(figLogPiTraceWarmUp,['./' dirname '/' subdirname '/' name '_logPiTraceX_warmup.png' ]);
        end
        
        figLogPiTrace=figure;
        plot(results.logPiTraceX(2:end),'r','LineWidth',1.5,'MarkerSize',8); hold on;
        title('\propto log p(X^n|y,\theta, \alpha) in SAPG algorithm');xlabel('Iteration (n)');grid on;hold off;
        saveas(figLogPiTrace,['./' dirname '/' subdirname '/' name '_logPiTraceX.png' ]);
    end
    if save_gx_plot
        figSum=figure;
        plot(results.gXTrace(1:results.last_samp-1),'r','LineWidth',1.5,'MarkerSize',8);hold on;
        plot(dimX*((results.thetas(1:results.last_samp-1)).^(-1)),'b','LineWidth',1.5,'MarkerSize',8);
        legend('g(x)= TV(x)','dimX/\theta_n');grid on; xlabel('Iteration (n)');hold off;
        saveas(figSum,['./' dirname '/' subdirname '/' name '_gX.png']);
    end
    
    %% Sigma^2
    figSigma=figure;
    sig = sigma^2 * ones(results.last_samp,1);
    plot(results.sigmas(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
    plot(sig, 'g-');
    xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('$\sigma^2_n$', 'Interpreter', 'latex');title('$\sigma^2_n$', 'Interpreter', 'latex'); 
    legend('$\sigma^2_n$', '$\sigma^2_{true}$', 'Interpreter', 'latex');hold off;
    saveas(figSigma,['./' dirname '/' subdirname '/' name '_sigma.png' ]);
    
    
    %% Theta
    figTheta=figure;
    plot(results.thetas(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);
    xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\theta_n');title('\theta_n')
    saveas(figTheta,['./' dirname '/' subdirname '/' name '_thetas.png' ]);
    
    %% tau
    to_vec = tau * ones(results.last_samp,1);
    figTau=figure;
    plot(results.taus(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
    plot(to_vec, 'g-');
    legend('\alpha_n', '\alpha = 0.4');hold off;
    xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\alpha_n');title('\alpha_n')
    saveas(figTau,['./' dirname '/' subdirname '/' name '_aphas.png' ]);
    
    %% tolerance plot to track convergence
    tol_line = op.stopTol * ones(1,results.last_samp);
    %% tau
    figtol_Alpha=figure;
    plot(results.tol_taus(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
    plot(tol_line,'r','LineWidth',1.5,'MarkerSize',8);
    xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('$\frac{\alpha_n-\alpha_{n-1}}{\alpha_{n-1}}$','Interpreter','latex');title('Tolerance w.r.t \alpha_n');hold off;
    saveas(figtol_Alpha,['./' dirname '/' subdirname '/' name '_tol_alphas.png' ]);
    
    %% theta
    figtol_Theta=figure;
    plot(results.tol_thetas(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
    plot(tol_line,'r','LineWidth',1.5,'MarkerSize',8);
    xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('$\frac{\theta_n-\theta_{n-1}}{\theta_{n-1}}$','Interpreter','latex');title('Tolerance w.r.t \theta_n');hold off;
    saveas(figtol_Theta,['./' dirname '/' subdirname '/' name '_tol_theta.png' ]);
    
    %% mean
    %theta
    figmean_Theta=figure;
    plot(results.mean_thetas(1:results.last_samp - op.burnIn),'b','LineWidth',1.5,'MarkerSize',8);
    xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\theta_n');title('Mean w.r.t \theta_n');legend('\theta_n');
    saveas(figmean_Theta,['./' dirname '/' subdirname '/' name '_mean_theta.png' ]);
    %alpha
    figmean_Alpha=figure;
    plot(results.mean_taus(1:results.last_samp - op.burnIn),'b','LineWidth',1.5,'MarkerSize',8);
    xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\alpha_n');title('Mean w.r.t \alpha_n');legend('$\alpha_n$', 'Interpreter', 'latex');
    saveas(figmean_Alpha,['./' dirname '/' subdirname '/' name '_mean_alpha.png' ]);
    %sigma^2
    figmean_Sigma=figure;
    plot(results.mean_sigmas(1:results.last_samp - op.burnIn),'b','LineWidth',1.5,'MarkerSize',8);
    xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('$\bar{\sigma^2_n}$', 'Interpreter', 'latex');title('$\bar{\sigma^2_n}$', 'Interpreter', 'latex');legend('$\sigma^2_n$', 'Interpreter', 'latex');
    saveas(figmean_Sigma,['./' dirname '/' subdirname '/' name '_mean_sigma.png' ]);
    
    close all;
    
end
