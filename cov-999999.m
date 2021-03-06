%% hamilton x2z adaptive potential
#rand('seed',54321);
#randn('seed',54321);
%% STEP SIZE

format long;

L = 5;
D = 2;

episode = 10000;
BURNIN0 = 1000;
BURNIN1 = 3000;
BURNIN2 = 6000;
INTERVAL = 100;

decay_factor = 1/exp(log(10)/BURNIN1);
KK = [0.1 0.5 0.9 0.99,0.999,0.9999,0.99999,0.999999,0.9999999];
dst =[];
for ii = 1:length(KK)
  delta = .001;
  decay = 0.01;
  SIGMA=[1,KK(ii);KK(ii),1];

  U = @(x) x' * inv(SIGMA) * x/2;
  dU = @(x) inv(SIGMA)*x;
  K = @(x) x' *  x/2;
  dK = @(x) x;
  ##
  xStar = randn(D,1);
  x0 = xStar;
  pStar = randn(D,1);
  z2x = @(z) z;
  x2z = @(x) x;
  E_total = U(xStar) + K(pStar);

  x = [xStar];
  p = [pStar];
  ##--
  data = [];
  energies = [];
  dst_sigma = [];
  dst_mu = [];
  Alpha = 1;
  n = 30;
  cov_hat = eye(2);
  cov_hat_half = cov_hat;
  for i= 1:episode
    xStar = x(:,end);
    if E_total > U(xStar);
      if i == BURNIN0
        ## bayesian
        mu_0 = mean(x,2);
        kappa_0 = BURNIN0;
        LAMBDA_0 = cov(x');
        nu_0 = BURNIN0;
      elseif i > BURNIN0  && mod(i,n) == 0 && i < BURNIN1
        y = x(:,end-n:end);
        ## bayesian
        y_bar = mean(y,2);
        S = (n-1) * cov(y');
        mu_n = kappa_0 * mu_0 / (kappa_0 + n) + n / (kappa_0 + n) * y_bar;
        kappa_n = kappa_0 + n;
        nu_n = nu_0 + n;
        LAMBDA_n = LAMBDA_0 + S + kappa_0 * n / (kappa_0 + n) * (y_bar - mu_0) * (y_bar - mu_0)';

        cov_hat = LAMBDA_n/(nu_0+n-D-1);
        #fprintf('%f\t%f\n',norm(cov_hat - SIGMA),norm(mu_0));
        mu_0 = mu_n;
        kappa_0 = kappa_n;
        nu_0 = nu_n;
        LAMBDA_0 = LAMBDA_n;
        cov_hat_half = sqrtm(cov_hat);
        z2x = @(z) cov_hat_half * z + mu_0;
        x2z = @(x) inv(cov_hat_half) * (x - mu_0);
        dst_sigma(:,end+1) = [i norm(cov_hat - SIGMA)];
        dst_mu(:,end+1) = [i norm(mu_0)];
      end
      target_k = E_total - U(xStar);
    else
      target_k = 1e-3;
    end
    R = sqrt(target_k*2);

    p0 = randn(D,1);
    pStar = p0 / norm(p0)*R;
    E_total = U(xStar) + K(pStar);
    zc = 0;
    E_t = [U(xStar)];
    zStar = x2z(xStar);
    for j= 1:L
      zStar = zStar + delta*dK(pStar);
      pStar = pStar - delta*cov_hat_half*dU(z2x(zStar));
      E_t(end+1) = U(z2x(zStar));
      if j >= 2 && zc == 0 
        if (E_t(end-2) - E_t(end-1))*(E_t(end-1) - E_t(end))<0 
          zc = j;
        end
      end
    end
    xStar = z2x(zStar);
    if mod(i,1000)==0
      fprintf('%f\t%f\t%f\t%f\t%f\t%d\t%d\n', U(xStar) , K(pStar),U(xStar)+K(pStar),E_total,delta,zc,L);
    end

    alpha = min(1,exp(U(x(:,end))- U(xStar)));
    Alpha = Alpha * 0.9 + alpha * 0.1;
    u = rand();
    if u > alpha
      x(:,end+1) = x(:,end);
      p(:,end+1) = p(:,end);
    else
      x(:,end+1) = xStar;
      p(:,end+1) = pStar;
    end
    if zc==L &&  alpha < rand()/2
      delta = delta / (1. + decay);
    elseif zc==0 && alpha > rand()/2 + .5
      delta = delta * (1. + decay);
      #end
    end

    if alpha > rand()/2 + .5
      E_total = E_total * (1. + decay);
    elseif alpha < rand()/2
      E_total = E_total / (1. + decay);
    end
    decay = decay * decay_factor;
    data(:,end+1) = [delta, E_total,Alpha];
    if i > 1
      energies(:,end+1) = [energies(1,end)*.9 + .1*U(xStar), energies(2,end)*.9 + .1*K(pStar),energies(3,end)*.9 + .1*(U(xStar)+K(pStar))];
    else
      energies(:,end+1) = [U(x(:,end)), K(p(:,end)),U(x(:,end))+K(p(:,end))];
    end
    ## LS(end+1) = L;
  end
  ##---------------------------------------------------------------------------------------------
  delta = .001;
  decay = 0.01;
  cov_hat = eye(2);
  cov_hat_half = eye(2);
  x1 = [x0];
  for i= 1:episode
    xStar = x1(:,end);
    if E_total > U(xStar);
      if i == BURNIN0
        ## bayesian
        mu_0 = mean(x1,2);
        kappa_0 = BURNIN0;
        LAMBDA_0 = cov(x1');
        nu_0 = BURNIN0;
      elseif i > BURNIN0  && mod(i,n) == 0 && i < BURNIN1
        y = x1(:,end-n:end);
        ## bayesian
        y_bar = mean(y,2);
        S = (n-1) * cov(y');
        mu_n = kappa_0 * mu_0 / (kappa_0 + n) + n / (kappa_0 + n) * y_bar;
        kappa_n = kappa_0 + n;
        nu_n = nu_0 + n;
        LAMBDA_n = LAMBDA_0 + S + kappa_0 * n / (kappa_0 + n) * (y_bar - mu_0) * (y_bar - mu_0)';

        cov_hat = LAMBDA_n/(nu_0+n-D-1);
        #vol = exp(sum(log(abs(eig(cov_hat))))/2);
        #cov_hat = cov_hat/vol;
        #fprintf('%f\t%f\n',norm(cov_hat - SIGMA),norm(mu_0));
        mu_0 = mu_n;
        kappa_0 = kappa_n;
        nu_0 = nu_n;
        LAMBDA_0 = LAMBDA_n;
        cov_hat_half = sqrtm(cov_hat);
        z2x = @(z) cov_hat_half * z + mu_0;
        x2z = @(x) inv(cov_hat_half) * (x - mu_0);
        dst_sigma(:,end+1) = [i norm(cov_hat - SIGMA)];
        dst_mu(:,end+1) = [i norm(mu_0)];
      end
      target_k = E_total - U(xStar);
    else
      target_k = 1e-3;
    end
    R = sqrt(target_k*2);

    pStar = randn(D,1);
    pStar = inv(cov_hat_half)*pStar;
    pStar = pStar / norm(pStar)*R;

    E_total = U(xStar) + K(pStar);
    zc = 0;
    for j= 1:L
      xStar = xStar + delta*cov_hat*dK(pStar);
      pStar = pStar - delta*dU(xStar);
      E_t(end+1) = U(xStar);
      if j >= 2 && zc == 0 
        if (E_t(end-2) - E_t(end-1))*(E_t(end-1) - E_t(end))<0 
          zc = j;
        end
      end
    end
    if mod(i,1000)==0
      fprintf('%f\t%f\t%f\t%f\t%f\t%d\t%d\n', U(xStar) , K(pStar),U(xStar)+K(pStar),E_total,delta,zc,L);
    end

    alpha = min(1,exp(U(x1(:,end))- U(xStar)));
    Alpha = Alpha * 0.9 + alpha * 0.1;
    u = rand();
    if u > alpha
      x1(:,end+1) = x1(:,end);
      p(:,end+1) = p(:,end);
    else
      x1(:,end+1) = xStar;
      p(:,end+1) = pStar;
    end
    if zc==L &&  alpha < rand()/2
      delta = delta / (1. + decay);
    elseif zc==0 && alpha > rand()/2 + .5
      delta = delta * (1. + decay);
      #end
    end

    if alpha > rand()/2 + .5
      E_total = E_total * (1. + decay);
    elseif alpha < rand()/2
      E_total = E_total / (1. + decay);
    end
    decay = decay * decay_factor;
    data(:,end+1) = [delta, E_total,Alpha];
    if i > 1
      energies(:,end+1) = [energies(1,end)*.9 + .1*U(xStar), energies(2,end)*.9 + .1*K(pStar),energies(3,end)*.9 + .1*(U(xStar)+K(pStar))];
    else
      energies(:,end+1) = [U(x(:,end)), K(p(:,end)),U(x(:,end))+K(p(:,end))];
    end
    ## LS(end+1) = L;
  end
  ##---------------------------------------------------------------------------------------------
  K = @(p) (transpose(p)*p)/2;
  dK = @(p) p;
  x2 = [x0];
  delta = 0.01;
  for i= 1:episode
    pStar = randn(2,1);
    p_ = pStar;
    for j = 1:L
      xStar = xStar + delta*dK(pStar);
      pStar = pStar - delta*dU(xStar);
    end
    alpha = min(1,exp(U(x2(:,end)) + K(p_) - U(xStar) - K(pStar)));
    u = rand();
    if u < alpha
      x2(:,end+1) = xStar;
    else
      x2(:,end+1) = x2(:,end);
    end
  end

  x3 = [x0];
  delta = 0.1;
  for i= 1:episode
    pStar = randn(2,1);
    p_ = pStar;
    for j = 1:L
      xStar = xStar + delta*dK(pStar);
      pStar = pStar - delta*dU(xStar);
    end
    alpha = min(1,exp(U(x3(:,end)) + K(p_) - U(xStar) - K(pStar)));
    u = rand();
    if u < alpha
      x3(:,end+1) = xStar;
    else
      x3(:,end+1) = x3(:,end);
    end
  end
  dst(end+1,:) = [norm(cov(x(1:2,BURNIN2:end)')-SIGMA,1) norm(cov(x1(1:2,BURNIN2:end)')-SIGMA,1) norm(cov(x2(1:2,BURNIN2:end)')-SIGMA,1) norm(cov(x3(1:2,BURNIN2:end)')-SIGMA,1)];
end
sum(isnan(dst),1)
x=1:length(KK)
plot(x,dst(:,1),'k-','linewidth',3,x,dst(:,2),'r*-','linewidth',1,x,dst(:,3),'bx--','linewidth',2,dst(:,4),'cx--','linewidth',2);
set(gca,'XTick',1:length(KK))
set(gca, 'xticklabel',{'0.1','0.5','0.9','0.99','0.999','0.9999','0.99999','0.999999','0.9999999'})
xlabel('K');
xlim([1 length(KK)]);
##ylabel('distance of est. and true cov.');
ylabel('|\Sigma_{est} \Sigma|');
legend('the proposed','RHMC','HMC, delta=0.01','HMC, delta=0.1');
dst
pause
