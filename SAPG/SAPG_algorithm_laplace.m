% Charlesquin Kemajou Mbakam: mbakamkemajou@gmail.org
% Heriot-Watt University
% date: Juin 2022

%% Maximum Marignal Likelihood estimation of parameters

function [theta_EB, b_EB, sigma_EB, results] = SAPG_algorithm_laplace(y, op)
    tic;
    %% Setup
    % initialisation for MYULA sampler
    if not(isfield(op, 'X0'))
        op.X0 = y; %by default we assume dim(x)=dim(y) if nothing is specified and use X0=y
    end
    % Stop criteria (relative change tolerance)
    if not(isfield(op, 'stopTol'))
        op.stopTol = -1; % if no stopTol was defined we set it to a negative value so it's ignored.
        % in this case the SAPG algo. will stop after op.samples iterations
    end

    if not(isfield(op, 'warmup'))
        op.warmup=100;       %use 100 by default
    end

    %% MYULA sampler
    dimX=numel(op.X0);
    warm_sample = op.warm_sample;
    warmupSteps=op.warmup;
    err_warm(1) = MSE(op.X0, op.x);
    err_warm(warmupSteps) = 0;

    lamb =  op.lambda;
    gam =  op.gamma;
    results.lambda = lamb;
    results.gamma = gam;

    %% SAPG algorithm
    total_iter=op.samples;

    % theta
    min_theta = op.min_th;
    max_theta = op.max_th;

    % b
    b_init = op.b_init;
    min_b  = op.min_b;
    max_b = op.max_b;


    % sigma
    sigma = op.sigma;
    sigma = sigma^2;
    sigma_init = op.sigma_init;
    min_sigma = min(op.sigma_min, op.sigma_max);
    max_sigma = max(op.sigma_min, op.sigma_max);

    % Step size
    delta = @(i) op.d_scale*( (i^(-op.d_exp)) /dimX );

    %% Functions related to Bayesian model
    proxG =op.proxG;% Proximal operator of non-smooth part
    logPi = op.logPi;% We use this scalar summary to monitor convergence
    gradF_b = op.grad_b;
    gradF = op.gradF;% Gradient of smooth part
    gradF_sigma = op.gradF_sigma;
    g = op.g;

    %% MYULA Warm-up
    X_wu = op.X0;
    if(warmupSteps > 0)
        % Run MYULA sampler with fix theta to warm up the markov chain
        fix_theta = op.th_init;
        fix_b = op.b_init;
        fix_sigma = op.sigma_init;
        logPiTrace_WU(op.warmup) = 0;

        % Run MYULA sampling from the posterior of X:
        proxGX_wu = proxG(X_wu, lamb, fix_theta);
        fprintf('Running Warm up...     \n');
        for ii = 2:warmupSteps
            %Sample from posterior with MYULA:
            X_wu =  X_wu + gam * (proxGX_wu-X_wu)/lamb - ...
                gam*gradF(X_wu, fix_b, fix_sigma) + sqrt(2*gam)*randn(size(X_wu));
            X_wu = abs(X_wu);
            proxGX_wu = proxG(X_wu,lamb,fix_theta);

            %Save current state to monitor convergence
                logPiTrace_WU(ii) = logPi(X_wu,fix_theta,  fix_b, fix_sigma);

            %Display Progress
            if mod(ii, round(op.warmup / 100)) == 0
                fprintf('\b\b\b\b\t%2d%%', round(ii / (op.warmup) * 100));
            end
        end
        results.logPiTrace_WU=logPiTrace_WU;
    end

    %% Run SAPG algorithm to estimate theta and b and sigma2

    % theta
    thetas(total_iter)=0;
    thetas(1)=op.th_init;

    % sigma
    sigmas(1) = sigma_init;
    sigmas(total_iter) = 0;

    % b
    bs(total_iter) = 0;
    bs(1) = b_init;

    % tolerance
    tol_thetas(total_iter) =0;
    tol_bs(total_iter) = 0;
    tol_sigmas(total_iter) = 0;

    % mean Value
    mean_bs(total_iter - op.burnIn) = 0;
    mean_thetas(total_iter - op.burnIn) = 0;
    mean_sigmas(total_iter - op.burnIn) = 0;

    % tracking
    err_sample(1) = MSE(X_wu, op.x);
    err_sample(total_iter) = 0;

    %%
    logPiTraceX(total_iter)=0;  % to monitor convergence
    gX(total_iter)=0;           % to monitor how the regularisation function evolves
    logPiTraceX(1)=logPi(X_wu,thetas(1), bs(1), sigmas(1));

    X = X_wu;
    proxGX = proxG(X,lamb,thetas(1));

    % Hyper parameters
    true_psf = psf_laplace(op.psf_size, op.b);
    app_psf = psf_laplace(op.psf_size, bs(1));
    err_psf(1) = l2(app_psf, true_psf);
    
    %% Constant to scale the step size
    c_theta =0.01;
    c_b = 100;
    c_sigma2 =10000;

    fprintf('\nRunning SAPG algorithm...  \n');
    for ii = 2:total_iter
        %% Sample from posterior with MYULA
        g_b(1) = 0;
        g_b(warm_sample) = 0;
        g_s(1) = 0;
        g_s(warm_sample) = 0;
        g_t(1) = 0;
        g_t(warm_sample) = 0;

        for jj = 1:1
            Z = randn(size(X));
            X =   X + gam * (proxGX-X)/lamb - gam*gradF(X, bs(ii-1), sigmas(ii-1)) + sqrt(2*gam)*Z;
            X = abs(X);
            proxGX = proxG(X,lamb,thetas(ii-1));

            g_b(jj) = gradF_b(X, bs(ii-1), sigmas(ii-1));
            g_s(jj) = gradF_sigma(X, bs(ii-1), sigmas(ii-1));
            g_t(jj) = (dimX/thetas(ii-1)- g(X));  %*exp(etas(ii-1));
        end

        G_b = mean(g_b);
        G_s = mean(g_s);
        G_t = mean(g_t);

        %% Update theta
          thetaii = thetas(ii-1) + c_theta* delta(ii)*G_t;
          thetas(ii) = min(max(thetaii,min_theta),max_theta);

        %% update b
          if op.fix_b
              bii = op.b;
          else
              bii = bs(ii-1) -  c_b*delta(ii) * G_b;
          end
          bs(ii) = min(max(bii ,min_b),max_b);

        %% update sigma
          if op.fix_sigma
              sigmaii = op.sigma^2;
          else
              sigmaii = sigmas(ii-1) + c_sigma2*delta(ii) * G_s;
          end
          sigmas(ii) = min(max(sigmaii,min_sigma),max_sigma);

        %% tracking
            err_sample(ii) = MSE(X, op.x);
            app_psf = psf_laplace(op.psf_size, bs(ii));
            err_psf(ii) = l2(app_psf, true_psf);

        %% Save current state to monitor convergence
            logPiTraceX(ii) = logPi(X,thetas(ii-1), bs(ii-1), sigmas(ii-1));
            gX(ii-1) = g(X);

        %% Display Progress
            if mod(ii, round(total_iter / 100)) == 0
                fprintf('\b\b\b\b\t%2d%%', round(ii / (total_iter) * 100));
            end

        %% Check stop criteria. If relative error is smaller than op.stopTol stop
        % tol for theta
          relErrTh1=abs(mean(thetas(op.burnIn:ii))-mean(thetas(op.burnIn:ii-1)))/mean(thetas(op.burnIn:ii-1));
          tol_thetas(ii) = relErrTh1;

        % tol for b
          relErrTh2=abs(mean(bs(op.burnIn:ii))-mean(bs(op.burnIn:ii-1)))/mean(bs(op.burnIn:ii-1));
          tol_bs(ii) = relErrTh2;

        % tol for sigma
        relErrTh4=abs(mean(sigmas(op.burnIn:ii))-mean(sigmas(op.burnIn:ii-1)))/mean(sigmas(op.burnIn:ii-1));
        tol_sigmas(ii) = relErrTh4;

        if ii > op.burnIn
            %% mean of theta
            mean_thetas(ii- op.burnIn)  = mean(thetas(op.burnIn:ii));
            %% mean b
            mean_bs(ii - op.burnIn) = mean(bs(op.burnIn:ii));
            %% mean sigma
            mean_sigmas(ii - op.burnIn) = mean(sigmas(op.burnIn:ii));

        end
    end

    %% Save logs in results struct
      results.execTimeFindTheta=toc;
      last_samp=ii;results.last_samp=last_samp;
      results.logPiTraceX=logPiTraceX(1:last_samp);
      results.gXTrace=gX(1:last_samp);

    %% theta
      theta_EB = mean(thetas(op.burnIn:last_samp));
      results.mean_theta= theta_EB;
      results.last_theta= thetas(last_samp);
      results.thetas=thetas(1:last_samp);
      results.mean_thetas = mean_thetas;
      results.tol_thetas = tol_thetas;
     results.c_theta = c_theta;

    %% b
      b_EB = mean(bs(op.burnIn:last_samp));
      results.mean_b= b_EB;
      b_last = bs(last_samp);
      results.last_b= b_last;
      results.bs = bs(1:last_samp);
      results.mean_bs = mean_bs;
      results.tol_bs = tol_bs;
      results.c_b = c_b;

    %% sigma
      sigma_EB = mean(sigmas(op.burnIn:last_samp));
      results.sigma_EB= sigma_EB;
      sigma_last = sigmas(last_samp);
      results.last_sigma= sigma_last;
      results.sigmas=sigmas(1:last_samp);
      results.mean_sigmas = mean_sigmas;
      results.tol_sigma = tol_sigmas;
      results.c_sigma2 = c_sigma2;

    %% final results
      results.X_sample = X;
      results.X_warm = X_wu;
      results.err_warm = err_warm;
      results.err_sample = err_sample;
      results.err_psf = err_psf;
      results.options=op;
end
