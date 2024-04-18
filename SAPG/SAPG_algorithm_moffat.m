% Charlesquin Kemajou Mbakam: mbakamkemajou@gmail.org
% Heriot-Watt University
% date: Juin 2022

%% Maximum Marignal Likelihood estimation of parameters

function [theta_EB, alpha_EB, beta_EB, sigma2_EB, results] = SAPG_algorithm_moffat(y, op)
    tic;
    % initialisation for MYULA sampler
    if not(isfield(op, 'X0'))
        op.X0 = y; %by default we assume dim(x)=dim(y) if nothing is specified and use X0=y
    end
    % Stop criteria (relative change tolerance)
    if not(isfield(op, 'stopTol'))
        op.stopTol = -1; % if no stopTol was defined we set it to a negative value so it's ignored.
    end

    %% MYULA sampler
    if not(isfield(op, 'warmup'))
        op.warmup=100;       %use 100 by default
    end

    dimX=numel(op.X0);
    warmupSteps=op.warmup;
    sub_sample = op.sub_sample;
    lamb  = op.lambda;
    gam =   op.gamma;
    results.lambda = lamb;
    results.gamma = gam;

    %% SAPG algorithm
      total_iter=op.samples;
    % theta
      min_theta=op.min_th;
      max_theta=op.max_th;

    % alpha
      alpha_init = op.alpha_init;
      min_alpha  = op.min_alpha;
      max_alpha = op.max_alpha;

    % beta
      beta_init = op.beta_init;
      min_beta  = op.min_beta;
      max_beta = op.max_beta;

    % sigma
      sigma_init = op.sigma_init;
      min_sigma = min(op.sigma_min, op.sigma_max);
      max_sigma = max(op.sigma_min, op.sigma_max);

    %% step size SAPG
      delta = @(i) op.d_scale*( (i^(-op.d_exp)) / dimX );

    %% Functions related to Bayesian model
      gradF = op.gradF;% Gradient of smooth part
      proxG =op.proxG;% Proximal operator of non-smooth part
      logPi = op.logPi;% We use this scalar summary to monitor convergence
      gradF_alpha = op.grad_alpha;
      gradF_beta = op.grad_beta;
      g = op.g;
      gradF_sigma = op.gradF_sigma;

    %% MYULA Warm-up
    X_wu = op.X0;
    if(warmupSteps > 0)
        % Run MYULA sampler with fix theta to warm up the markov chain
          fix_theta = op.th_init;
          fix_alpha = alpha_init;
          fix_beta = beta_init;
          fix_sigma = sigma_init;
          logPiTrace_WU(op.warmup) = 0;%logPiTrace to monitor convergence

        % Run MYULA sampling from the posterior of X:
          proxGX_wu = proxG(X_wu, lamb, fix_theta);
          fprintf('Running Warm up     \n\n');
        for ii = 2:warmupSteps
            % Sample from posterior with MYULA:

              X_wu =  X_wu + gam * (proxGX_wu-X_wu)/lamb - gam * gradF(X_wu, fix_alpha, fix_beta, fix_sigma) ...
                  + sqrt(2*gam)*randn(size(X_wu));
              X_wu = abs(X_wu);
              proxGX_wu = proxG(X_wu,lamb,fix_theta);

            % Save current state to monitor convergence0.1*
              logPiTrace_WU(ii) = logPi(X_wu,fix_theta, fix_alpha, fix_beta, fix_sigma);

            %Display Progress
            if mod(ii, round(op.warmup / 100)) == 0
                fprintf('\b\b\b\b\t%2d%%', round(ii / (op.warmup) * 100));
            end
        end
        results.logPiTrace_WU=logPiTrace_WU;
    end

    %% Run SAPG algorithm to estimate execTimeFindParameters
    % theta
      thetas(total_iter)=0;
      thetas(1)=op.th_init;

    % sigma
      sigmas(1) = sigma_init;
      sigmas(total_iter) = 0;

    % alpha
      alphas(total_iter) = 0;
      alphas(1) = alpha_init;

    % beta
      betas(total_iter) = 0;
      betas(1) = beta_init;

    % tolerance
      tol_thetas(total_iter) =0;
      tol_alphas(total_iter) = 0;
      tol_betas(total_iter) = 0;
      tol_sigmas(total_iter) = 0;

    % mean Value
      mean_alphas(total_iter - op.burnIn) = 0;
      mean_betas(total_iter - op.burnIn) = 0;
      mean_thetas(total_iter - op.burnIn) = 0;
      mean_sigmas(total_iter - op.burnIn) = 0;

    %%
      logPiTraceX(total_iter)=0;% to monitor convergence
      gX(total_iter)=0; % to monitor how the regularisation function evolves
      logPiTraceX(1)=logPi(X_wu,thetas(1), alphas(1), betas(1), sigmas(1));

      X = X_wu;% start MYULA markov chain from last sample after warmup
      proxGX = proxG(X,lamb,thetas(1));


    %% Constant
      c_theta = 0.1;
      c_alpha = 10;
      c_beta = 10000;
      c_sigma2 = 10000;

    fprintf('\nRunning SAPG algorithm     \n');
    for ii = 2:total_iter
        %% Sample from posterior with MYULA:
        g_b(1) = 0;
        g_b(sub_sample) = 0;
        g_a(1) = 0;
        g_a(sub_sample) = 0;
        g_s(1) = 0;
        g_s(sub_sample) = 0;
        g_t(1) = 0;
        g_t(sub_sample) = 0;

        %% distance betwen true and approximate psf
        psf_size = op.psf_size;
        true_psf = psf_moffat(psf_size, op.alpha, op.beta);
        app_psf  = psf_moffat(psf_size, op.alpha_init, beta_init);
        psf_err(1) = l2(true_psf, app_psf);

        for jj = 1:1
            Z = randn(size(X));
            X = X + gam * (proxGX-X)/lamb - gam*gradF(X, alphas(ii-1), betas(ii-1), sigmas(ii-1)) + sqrt(2*gam)*Z;
            X = abs(X);

            proxGX = proxG(X,lamb,thetas(ii-1));
            g_b(jj) = gradF_beta(X, alphas(ii-1), betas(ii-1), sigmas(ii-1));
            g_a(jj) = gradF_alpha(X, alphas(ii-1), betas(ii-1), sigmas(ii-1));
            g_s(jj) = gradF_sigma(X, alphas(ii-1), betas(ii-1), sigmas(ii-1));
            g_t(jj) = (dimX/thetas(ii-1)- g(X));
        end

        G_b = mean(g_b);
        G_s = mean(g_s);
        G_t = mean(g_t);
        G_a = mean(g_a);

        % Update theta
          thetaii = thetas(ii-1)  +   c_theta*delta(ii) * G_t;
          thetas(ii) = min(max(thetaii,min_theta),max_theta);

        % update alpha
          if op.fix_alpha
              alphaii = op.alpha;
          else
              alphaii = alphas(ii-1) - c_alpha*delta(ii) * G_a;
          end
          alphas(ii) = min(max(alphaii ,min_alpha),max_alpha);

        % update beta
          if op.fix_beta
              betaii = op.beta;
          else
              betaii = betas(ii-1) -  c_beta*delta(ii) * G_b;
          end
          betas(ii) = min(max(betaii,min_beta),max_beta);

        % update sigma
          if op.fix_sigma
              sigmaii = op.sigma^2;
          else
              sigmaii = sigmas(ii-1) + c_sigma2*delta(ii) * G_s ;
          end
          sigmas(ii) = min(max(sigmaii,min_sigma),max_sigma);
        
        %% tracking
        app_psf = psf_moffat(psf_size, alphas(ii), betas(ii));
        err_psf(ii) = l2(app_psf, true_psf);
        
        %% Save current state to monitor convergence
        logPiTraceX(ii) = logPi(X,thetas(ii-1), alphas(ii-1), betas(ii-1), sigmas(ii-1));
        gX(ii-1) = g(X);

        %% Display Progress
        if mod(ii, round(total_iter / 100)) == 0
            fprintf('\b\b\b\b%2d%%', round(ii / (total_iter) * 100));
        end

        %% Check stop criteria. If relative error is smaller than op.stopTol stop
        % tol for theta
          relErrTh1=abs(mean(thetas(op.burnIn:ii))-mean(thetas(op.burnIn:ii-1)))/mean(thetas(op.burnIn:ii-1));
          tol_thetas(ii) = relErrTh1;

        % tol for alpha
          relErrTh2=abs(mean(alphas(op.burnIn:ii))-mean(alphas(op.burnIn:ii-1)))/mean(alphas(op.burnIn:ii-1));
          tol_alphas(ii) = relErrTh2;

        % tol for beta
          relErrTh3=abs(mean(betas(op.burnIn:ii))-mean(betas(op.burnIn:ii-1)))/mean(betas(op.burnIn:ii-1));
          tol_betas(ii) = relErrTh3;

        % tol for sigma
          relErrTh4=abs(mean(sigmas(op.burnIn:ii))-mean(sigmas(op.burnIn:ii-1)))/mean(sigmas(op.burnIn:ii-1));
          tol_sigmas(ii) = relErrTh4;

        if ii > op.burnIn
            % mean of theta
              mean_thetas(ii- op.burnIn)  = mean(thetas(op.burnIn:ii));
            % mean alpha
              mean_alphas(ii - op.burnIn) = mean(alphas(op.burnIn:ii));
            % mean beta
              mean_betas(ii - op.burnIn) = mean(betas(op.burnIn:ii));
            % mean sigma
              mean_sigmas(ii - op.burnIn) = mean(sigmas(op.burnIn:ii));
        end

    end

    % Save logs in results struct
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


    %% alpha
      alpha_EB = mean(alphas(op.burnIn:last_samp));
      results.alpha_EB= alpha_EB;
      alpha_last = alphas(last_samp);
      results.last_alpha= alpha_last;
      results.alphas = alphas(1:last_samp);
      results.mean_alphas = mean_alphas;
      results.tol_alphas = tol_alphas;
      results.c_alpha = c_alpha;

    %% beta
      beta_EB = mean(betas(op.burnIn:last_samp));
      results.beta_EB = beta_EB;
      beta_last = betas(last_samp);
      results.last_beta = beta_last;
      results.betas = betas(1:last_samp);
      results.mean_betas = mean_betas;
      results.tol_betas = tol_betas;
      results.c_beta = c_beta;

    %% sigma
      sigma2_EB = mean(sigmas(op.burnIn:last_samp));
      results.sigma_EB= sigma2_EB;
      sigma_last = sigmas(last_samp);
      results.last_sigma= sigma_last;
      results.sigmas=sigmas(1:last_samp);
      results.mean_sigmas = mean_sigmas;
      results.tol_sigma = tol_sigmas;
      results.c_sigma2 = c_sigma2;

    %% final results
      results.Xlast_sample = X;
      results.X_warm = X_wu;
      results.options=op;
      results.err_psf = err_psf;
end
