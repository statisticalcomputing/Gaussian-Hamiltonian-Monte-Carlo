function [x, p] = ghmc(U,dU, D, x0, EPISODE=10000, BURNIN0=100,BURNIN1=3000,BURNIN2=5000,INTERVAL = 100,n = 30, L=5,decay=0.01, delta=0.001)


decay_factor = 1/exp(log(10)/BURNIN1);
K = @(x) x' *  x/2;
dK = @(x) x;
##
if ~x0
  x0 = argmin(U,dU, randn(D,1));
end
xStar = x0;
pStar = randn(D,1);
z2x = @(z) z;
x2z = @(x) x;
E_total = U(xStar) + K(pStar);

x = [xStar];
p = [pStar];
##--


cov_hat = eye(2);
cov_hat_half = cov_hat;
for i= 1:EPISODE
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
end
