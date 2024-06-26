 %% Experiment 4.2.1 - Deblurring with Total-Variation prior: Guassian Kernel with unknown bandwidth.

%% Create array with test images names
clear all;clc;
testImg = dir('images');
testImg = {testImg(3:end).name};
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
op.samples = 5000; % max iterations for SAPG algorithm to estimate theta
op.stopTol = 1e-4; % tolerance in relative change of theta_EB to stop the algorithm
op.tol2 = 1e-2;  % tolerance for tau
op.burnIn =100;	% iterations we ignore before taking the average over iterates theta_n

op.min_th = 1e-3; % projection interval Theta (min theta)
op.max_th = 1; % projection interval Theta (max theta)
op.min_to = 0.01;
op.max_to = 1;

tau = 0.4;
op.th_init = 0.01; % theta_0 initialisation of the SAPG algorithm
op.to_init = 0.7; % tau_0 initialization of the SAPG algorithm

% delta(i) for SAPG algorithm defined as: op.d_scale*( (i^(-op.d_exp)) / numel(x) );
op.d_exp =  0.8;
op.d_scale =  0.01/op.th_init;

%MYULA parameters
op.warmup = 3000 ; % number of warm-up iterations with fixed theta for MYULA sampler
op.lambdaMax = 2; % max smoothing parameter for MYULA
lambdaFun= @(Lf) min((5/Lf),op.lambdaMax);
gamma_max = @(Lf,lambda) 1/(Lf+(1/lambda));%Lf is the lipschitz constant of the gradient of the smooth part
op.gammaFrac=0.98; % we set gamma=op.gammaFrac*gamma_max

randn('state',1); % Set rnd generators (for repeatability)
%% MSE
Theta_op = 5e-4:0.005:4e-1;%[1.7e-3 1e-2 1.2e-2 1.5e-2 1.8e-2 2e-2 2.5e-2 3e-2 4e-2 5e-2 6e-2];
Tau_op = 0.1:0.01:0.8;%[0.3 0.4 0.5 0.6 0.7 0.8];
MSE_theta(length(Theta_op)) = 0;
MSE_tau(length(Tau_op)) = 0;
%% RUN EXPERIMENTS
for i_im = 10:10%length(testImg)
    for bsnr= [30]
        fprintf('SNR=%d\n',bsnr); op.BSNRdb=bsnr;
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
        sigma = norm(Ax-mean(mean(Ax)),'fro')/sqrt(dimX*10^(op.BSNRdb/10));
        sigma2 = sigma^2; 
        op.sigma=sigma;
        y = Ax + sigma * randn(size(Ax));
        op.X0 = y;
        
        %% Experiment setup: functions related to Bayesian model
        % Regulariser
        % TV norm
        op.g = @(x) TVnorm(x); %g(x) for TV regularizer
        chambolleit = 25;
        
        % Proximal operator of g(x)         
        op.proxG = @(x,lambda,theta) chambolle_prox_TV_stop(x,'lambda',lambda*theta,'maxiter',chambolleit);
        Psi = @(x,th) op.proxG(x,1,th); % define this format for SALSA solver
                         
        %% Likelihood (data fidelity)
        op.f = @(x, tau) (norm(y-A(x, tau),'fro')^2)/(2*sigma2); % p(y|x)∝ exp{-op.f(x)}
        op.gradF = @(x, tau) real(AT(A(x, tau) - y, tau)/sigma2);  % Gradient of smooth part f
        op.grad_t = @(x, tau) real(sum(sum((dif_A(x, tau)) .* (A(x, tau) - y))) / sigma^2);
        Lf = (evMax/sigma)^2; % define Lipschitz constant of gradient of smooth part

        % We use this scalar summary to monitor convergence
        op.logPi = @(x,theta, tau) -op.f(x, tau) -theta*op.g(x);
        
        %% Set algorithm parameters that depend on Lf
        op.lambda=lambdaFun(Lf);%smoothing parameter for MYULA sampler
        op.gamma=op.gammaFrac*gamma_max(Lf,op.lambda);%discretisation step MYULA
        
        %% Run SAPG Algorithm 1 to compute theta_EB
        tic;
        [theta_EB, tau_EB, results] = SAPG_algorithm_1(y, op);
        %[theta_EB, tau_EB, results] = SAPG_tau_theta(y, op);%
        %[theta_EB, tau_EB, results] = SAPG_theta_tau(y, op);
        sapg_time = toc;
        results.SAPG_time = sapg_time;
        op.theta_EB = theta_EB;
        op.tau_EB = tau_EB;
        last_tau = results.last_tau;
                fprintf('Theta optimal: %d\t Tau optimal: %d\t Last tau: %d', theta_EB, tau_EB, last_tau);
        %% Solve MYULA problem with theta_op and tau_op       
        global calls;
        calls = 0;
        %A = @(x) callcounter(A,x);
        %AT = @(x) callcounter(AT,x);
        
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
            SALSA_v2(y, A1, theta_EB*sigma^2,...
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
        results.mse=mse;
        results.xMAP=xMAP;
        results.x=x;
        results.y=y;
        
        %% last tau
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
            SALSA_v2(y, A1, theta_EB*sigma^2,...
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
        results.mse_last=mse_last;
        results.xMAP_last=xMAP_last;
        
        %% tau_true
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
            SALSA_v2(y, A_, theta_EB*sigma^2,...
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
        results.mse_true_tau = mse_true_tau;
        if 2==3
        %% fix theta
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
                SALSA_v2(y, A2, theta*sigma^2,...
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
        
        results.MSE_tau=MSE_tau;
        end
        
        %% fix tau
        for jj = 1:length(Theta_op)
            to = tau_EB;
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
                SALSA_v2(y, A2, theta*sigma^2,...
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
        salsa_time = toc;
        results.salsa_time = salsa_time;
        
        results.MSE_theta=MSE_theta;
        [m1, n1] = min(MSE_theta(:));
        x_oracle = Theta_op(n1);
        y_oracle = m1;
        
        %% Save Results
        %Create directories
        %subdirID: creates aditional folder with this name to store results
        subdirID = ['algo1_tol' char(num2str(op.stopTol)) '_maxIter' char(num2str(op.samples)) ];
        dirname = 'results_theta_tau_join/4.2.1-deblur-TV';
        subdirname=['BSNR' char(num2str(op.BSNRdb)) '/' subdirID ];
        subdirname  = strcat(fix, subdirname);
        if ~exist(['./' dirname ],'dir')
            mkdir('results_theta_tau_join');
        end
        if ~exist(['./' dirname '/' subdirname ],'dir')
            mkdir(['./' dirname '/' subdirname ]);
        end
        
        % Save data
        [~,name,~] = fileparts(char(filename));
        
        save(['./' dirname '/' subdirname '/' name '_results.mat'], '-struct','results');
        close all;
        if 2==3
        %% tau
        figMSE_tau=figure;
        plot(Tau_op, results.MSE_tau, 'b');hold on;
        plot( tau_EB, results.mse, '+', 'LineWidth',2.5, 'MarkerSize',8);
        plot( tau, results.mse_true_tau, '*' , 'LineWidth',2.5, 'MarkerSize',8);
        plot( last_tau,results.mse_last, 'o');
        title('MSE \alpha');xlabel('\alpha'); 
        legend('MSE','\alpha_{mean}',  '\alpha_{oracle}', '\alpha_N');hold off;
        saveas(figMSE_tau,['./' dirname '/' subdirname '/' name '_MSE_alpha.png']);
        end
        
        %% theta
        figMSE_theta=figure;
        plot(Theta_op, results.MSE_theta, 'b');hold on;
        plot( theta_EB, results.mse, 'o', 'LineWidth',2.5, 'MarkerSize',8);
        plot( x_oracle, y_oracle, '*', 'LineWidth',2.5, 'MarkerSize',8);
        title('MSE \theta');xlabel('\theta'); 
        legend('MSE','\theta_{mean}', '\theta_{oracle}');hold off;
        saveas(figMSE_theta,['./' dirname '/' subdirname '/' name '_MSE_theta.png']);
        
        
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
        
        %% Theta
        figTheta=figure;
        plot(results.thetas(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);
        xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\theta_n');title('\theta_n');
        saveas(figTheta,['./' dirname '/' subdirname '/' name '_thetas.png' ]);
        
        %% tau
        t = tau * ones(results.last_samp, 1);
        figTau=figure;
        plot(results.taus(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
        plot(t, 'g' ,'LineWidth',1.5,'MarkerSize',8);
        legend('\alpha_n','True \alpha');
        xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\alpha_n');title('\alpha_n');hold off;
        saveas(figTau,['./' dirname '/' subdirname '/' name '_aphas.png' ]);
        
        %% tolerance plot to track convergence
        tol_line = op.stopTol * ones(1,results.last_samp);
        %% tau
        figtol_Alpha=figure;
        plot(results.tol_taus(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
        plot(tol_line,'r','LineWidth',1.5,'MarkerSize',8);
        xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\alpha_n');title('Tolerance w.r.t \alpha_n');hold off;
        saveas(figtol_Alpha,['./' dirname '/' subdirname '/' name '_tol_alphas.png' ]);
        
        %% theta
        figtol_Theta=figure;
        plot(results.tol_thetas(1:results.last_samp),'b','LineWidth',1.5,'MarkerSize',8);hold on;
        plot(tol_line,'r','LineWidth',1.5,'MarkerSize',8);
        xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\theta_n');title('Tolerance w.r.t \theta_n');hold off;
        saveas(figtol_Theta,['./' dirname '/' subdirname '/' name '_tol_theta.png' ]);
        
        %% mean 
        figmean_Theta=figure;
        plot(results.mean_thetas(1:results.last_samp - op.burnIn),'b','LineWidth',1.5,'MarkerSize',8);
        xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\theta_n');title('Mean w.r.t \theta_n');
        saveas(figmean_Theta,['./' dirname '/' subdirname '/' name '_mean_theta.png' ]);
        
        figmean_Alpha=figure;
        plot(results.mean_taus(1:results.last_samp - op.burnIn),'b','LineWidth',1.5,'MarkerSize',8);
        xlim([1 inf]); grid on; xlabel('Iteration (n)');ylabel('\alpha_n');title('Mean w.r.t \alpha_n');
        saveas(figmean_Alpha,['./' dirname '/' subdirname '/' name '_mean_alpha.png' ]);
        
        close all;
    end
end