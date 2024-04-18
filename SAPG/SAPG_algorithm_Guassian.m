% Charlesquin Kemajou Mbakam: mbakamkemajou@gmail.org
% Heriot-Watt University
% date: Juin 2022

%% Maximum Marignal Likelihood estimation of parameters

function [theta_EB, w1_EB, w2_EB, sigma_EB, results] = SAPG_algorithm_sigma(y, op, c)
    tic;
    %% Setup
    if not(isfield(op, 'X0'))
        op.X0 = y; %by default we assume dim(x)=dim(y) if nothing is specified and use X0=y
    end
    if not(isfield(op, 'stopTol'))
        op.stopTol = -1; % if no stopTol was defined we set it to a negative value so it's ignored.
    end
    dimX=numel(op.X0);

    %% MYULA sampler
    if not(isfield(op, 'warmup'))
        op.warmup=100;       %use 100 by default
    end

    %% Gaussian kernel
    phi= op.phi;
    taille = op.psf_size;

    %% SAPG algorithm
    total_iter=op.samples;
    warmupSteps=op.warmup;
    lamb = c.lam*op.lambda;
    gam = c.gam*op.gamma;

    %% theta : regularisation parameter
    eta_init=log(op.th_init);
    min_theta=op.min_th;
    max_theta=op.max_th;

    %% w1 : horizontal bandwidth
    w1_init = op.w1_init;
    min_w1  = op.min_w1;
    max_w1 = op.max_w1;

    %% w2: vertical bandwidth
    w2_init = op.w2_init;
    min_w2  = op.min_w2;
    max_w2 = op.max_w2;

    %% sigma : Noise variance
    sigma = op.sigma^2;
    sigma_init = op.sigma_init;
    min_sigma = min(op.sigma_min, op.sigma_max);
    max_sigma = max(op.sigma_min, op.sigma_max);

    %% step size of the SAPG
    delta = @(i) op.d_scale*( (i^(-op.d_exp)) /dimX );

    %% Functions related to Bayesian model
    gradF = op.gradF;           % Gradient of smooth part
    proxG =op.proxG;            % Proximal operator of non-smooth part
    logPi = op.logPi;           % We use this scalar summary to monitor convergence
    gradF_w1 = op.grad_w1;
    gradF_w2 = op.grad_w2;
    g = op.g;
    gradF_sigma = op.gradF_sigma;

    %% MYULA Warm-up
    X_wu = op.X0;
    if(warmupSteps > 0)
        % Run MYULA sampler with fix theta to warm up the markov chain
        fix_theta = op.th_init;
        fix_w1 = op.w1_init;
        fix_w2 = op.w2_init;
        fix_sigma = sigma_init;
        logPiTrace_WU(op.warmup) = 0;   %logPiTrace to monitor convergence

        proxGX_wu = proxG(X_wu, fix_theta);
        fprintf('\nRunning Warm up...     \n');
        for ii = 2:warmupSteps
            %Sample from posterior with MYULA:
            X_wu =  abs(X_wu + gam* (proxGX_wu-X_wu)/lamb ...
                - gam*gradF(X_wu, fix_w1, fix_w2, fix_sigma) + sqrt(2*gam)*randn(size(X_wu)));
            proxGX_wu = proxG(X_wu,fix_theta);

            %Save current state to monitor convergence
            logPiTrace_WU(ii) = logPi(X_wu,fix_theta, fix_w1, fix_w2, fix_sigma);

            %Display Progress
            if mod(ii, round(op.warmup / 100)) == 0
                fprintf('\b\b\b\b\t%2d%%', round(ii / (op.warmup) * 100));
            end
        end
        results.logPiTrace_WU=logPiTrace_WU;
    end

    % ======================================================== %
    %% Run SAPG algorithm to estimate parameters
    % ======================================================== %

    %% theta
    thetas(total_iter)=0;
    thetas(1)=op.th_init;
   

    %% sigma
    sigmas(1) = sigma_init;
    sigmas(total_iter) = 0;

    %% w1
    w1s(total_iter) = 0;
    w1s(1) = w1_init;

     %% w2
    w2s(total_iter) = 0;
    w2s(1) = w2_init;

    %% tolerance
    tol_thetas(total_iter) =0;
    tol_w1s(total_iter) = 0;
    tol_w2s(total_iter) = 0;
    tol_sigma(total_iter) = 0;

    %% mean Value
    mean_w1s = zeros(total_iter - op.burnIn,1);
    mean_w2s = zeros(total_iter - op.burnIn,1);
    mean_thetas = zeros(total_iter - op.burnIn,1);
    mean_sigmas = zeros(total_iter - op.burnIn,1);

    %% Gradients
    Grad_thetas(1) = 0;
    Grad_w1(1) = 0;
    Grad_w2(1) = 0;
    Grad_sigma(1) = 0;

    %% monitor convergence
    logPiTraceX(total_iter)=0;
    gX(total_iter)=0; % to monitor how the regularisation function evolves
    logPiTraceX(1)=logPi(X_wu,thetas(1), w1s(1), w2s(1), sigmas(1));
    
    X = X_wu;% start MYULA markov chain from last sample after warmup
    proxGX = proxG(X,thetas(1));


    %%% Monitoring PSF convergence
    psf_true = psf_gaussian(taille, op.w1, op.w2, phi);
    psf_init = psf_gaussian(taille, w1s(1), w2s(1), phi);
    err_psf(1) = l2(psf_init, psf_true);

    %%% Monitoring step-size of the SAPG
    c_theta = c.theta;
    c_sigma = c.sigma;
    c_w1 = c.w1;
    c_w2 = c.w2;

    %%% Main loop for the SAPG

    fprintf('\nRunning SAPG algorithm...     \n');

    for ii = 2:total_iter
        %% Sample from posterior with MYULA:
        Z = randn(size(X));
        X =  abs(X + gam* (proxGX-X)/lamb - gam*gradF(X, w1s(ii-1), w2s(ii-1), sigmas(ii-1)) + sqrt(2*gam)*Z);
        proxGX = proxG(X,thetas(ii-1));

        %% Update theta
        G_t = (dimX/thetas(ii-1) - g(X));
        thetaii = thetas(ii-1) +  c_theta * delta(ii) * G_t;
        thetas(ii) = min(max(thetaii,min_theta),max_theta);

        %% update w1
        G_w1 = gradF_w1(X, w1s(ii-1), w2s(ii-1), sigmas(ii-1));
        if op.fix_w1
            w1ii = op.w1;
        else
            w1ii = w1s(ii-1) -  c_w1 * delta(ii) * G_w1;
        end
        w1s(ii) = min(max(w1ii ,min_w1),max_w1);

        %% update w2
        G_w2 = gradF_w2(X, w1s(ii-1), w2s(ii-1), sigmas(ii-1));
        if op.fix_w2
            w2ii = op.w2;
        else
            w2ii = w2s(ii-1) -  c_w2 * delta(ii) * G_w2;
        end
        w2s(ii) = min(max(w2ii ,min_w2),max_w2);

        %% update sigma
        G_s = gradF_sigma(X, w1s(ii-1), w2s(ii-1), sigmas(ii-1)) ;
        if op.fix_sigma
            sigmaii = op.sigma_init;
        else
            sigmaii = sigmas(ii-1) + c_sigma * delta(ii) * G_s;
        end
        sigmas(ii) = min(max(sigmaii,min_sigma),max_sigma);

        %% Graadient tracking
        Grad_sigma(ii) = G_s;
        Grad_w1(ii) = G_w1;
        Grad_w2(ii) = G_w2;
        Grad_theta(ii) = G_t;

        %% PSF tracking
        psf_appii = psf_gaussian(taille, w1s(ii), w2s(ii-1), phi);
        err_psf(ii) =  l2(psf_appii, psf_true);

        %% Save current state to monitor convergence
        logPiTraceX(ii) = logPi(X,thetas(ii-1), w1s(ii-1), w2s(ii-1), sigmas(ii-1));
        gX(ii-1) = g(X);

        %% Display Progress
        if mod(ii, round(total_iter / 100)) == 0
            fprintf('\b\b\b\b\t%2d%%', round(ii / (total_iter) * 100));
        end

        %% Check stop criteria. If relative error is smaller than op.stopTol stop

        % tol for theta
        relErrTh1=abs(mean(thetas(op.burnIn:ii))-mean(thetas(op.burnIn:ii-1)))/mean(thetas(op.burnIn:ii-1));
        tol_thetas(ii) = relErrTh1;

        % tol for w1
        relErrTh2=abs(mean(w1s(op.burnIn:ii))-mean(w1s(op.burnIn:ii-1)))/mean(w1s(op.burnIn:ii-1));
        tol_w1s(ii) = relErrTh2;

        % tol for tau
        relErrTh3=abs(mean(w2s(op.burnIn:ii))-mean(w2s(op.burnIn:ii-1)))/mean(w2s(op.burnIn:ii-1));
        tol_w2s(ii) = relErrTh3;

        % tol for sigma
        relErrTh4=abs(mean(sigmas(op.burnIn:ii))-mean(sigmas(op.burnIn:ii-1)))/mean(sigmas(op.burnIn:ii-1));
        tol_sigma(ii) = relErrTh4;

        % if ii == op.burnIn
        %     posterior_meanvar = weldford(X, 1);
        % end
        if ii > op.burnIn
            %% mean of theta
            mean_thetas(ii- op.burnIn)  = mean(thetas(op.burnIn:ii));
            %% mean w1
            mean_w1s(ii - op.burnIn) = mean(w1s(op.burnIn:ii));
            %% mean w2
            mean_w2s(ii - op.burnIn) = mean(w2s(op.burnIn:ii));
            % mean sigma
            mean_sigmas(ii - op.burnIn) = mean(sigmas(op.burnIn:ii));

            % posterior_meanvar.updated(X);
        end
    end

    %% Save logs in results struct
    results.execTimeFindParameters=toc;
    last_samp=ii;
    results.last_samp=last_samp;
    results.logPiTraceX=logPiTraceX(1:last_samp);
    results.gXTrace=gX(1:last_samp);

    %% theta
    theta_EB = mean(thetas(op.burnIn:last_samp));
    results.theta_EB= theta_EB;
    results.last_theta= thetas(last_samp);
    results.thetas=thetas(1:last_samp);
    results.mean_thetas = mean_thetas;
    results.tol_thetas = tol_thetas;

    %% w1
    w1_EB = mean(w1s(op.burnIn:last_samp));
    results.w1_EB= w1_EB;
    w1_last = w1s(last_samp);
    results.last_w1= w1_last;
    results.w1s = w1s(1:last_samp);
    results.mean_w1s = mean_w1s;
    results.tol_w1s = tol_w1s;

    %% w2
    w2_EB = mean(w2s(op.burnIn:last_samp));
    results.w2_EB= w2_EB;
    w2_last = w2s(last_samp);
    results.last_w2= w2_last;
    results.w2s = w2s(1:last_samp);
    results.mean_w2s = mean_w2s;
    results.tol_w2s = tol_w2s;

    %% sigma
    sigma_EB = mean(sigmas(op.burnIn:last_samp));
    results.sigma_EB= sigma_EB;
    sigma_last = sigmas(last_samp);
    results.last_sigma= sigma_last;
    results.sigmas=sigmas(1:last_samp);
    results.mean_sigmas = mean_sigmas;
    results.tol_sigma = tol_sigma;

    % results.posteriormean = posterior_meanvar.eval_mean();
    % results.posteriorvar = posterior_meanvar.eval_var();

    %% final results
    results.c_theta = c_theta;
    results.c_w1 = c_w1;
    results.c_w2 = c_w2;
    results.Xlast_sample = X;
    results.c_sigma = c_sigma;
    results.err_psf = err_psf;
    results.grad_theta = Grad_theta;
    results.grad_w1 = Grad_w1;
    results.grad_w2 = Grad_w2;
    results.grad_sigma = Grad_sigma;
    results.options=op;

end
