%% sample a bi-variate cov :[1,.9999;.9999,1]
#rand('seed',54321);
#randn('seed',54321);
%% STEP SIZE

format long;
delta = .0001;
L = 5;
D = 2;

episode = 10000;
BURNIN0 = 100;
BURNIN1 = 2000;
BURNIN2 = 4000;

decay_factor = 1/exp(log(1000)/BURNIN2);
decay = 0.1;
SIGMA=[1,.9999;.9999,1];
%%SIGMA=[99,1/99;1/99,99];

U = @(x) x' * inv(SIGMA) * x/2;
dU = @(x) inv(SIGMA)*x;
K = @(x) x' *  x/2;
dK = @(x) x;
##
xStar = randn(D,1);
pStar = randn(D,1);
z2x = @(z) z;
x2z = @(x) x;
E_total = U(xStar) + K(pStar);

x(:,end+1) = xStar;
p(:,end+1) = pStar;
##--
data = [];
dst =[];
rho  = .5;
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
  E_t = [K(pStar)];
  zStar = x2z(xStar);
  pStar = pStar - .5* delta*dU(z2x(zStar));
  for j= 1:L
    zStar = zStar + delta*dK(pStar);
    pStar = pStar - delta*cov_hat_half*dU(z2x(zStar));
    E_t(end+1) =K(pStar);# U(z2x(zStar));
    if j >= 2 && zc == 0 
      if (E_t(end-2) - E_t(end-1))*(E_t(end-1) - E_t(end))<0 
        zc = j;
      end
    end
  end
  pStar = pStar + .5* delta*dU(z2x(zStar));
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
  if i < BURNIN2
    if rand()<rho
      if (zc ==0 || zc == L)%% && i < BURNIN1
        if alpha > rand()*.7 + .3 && zc == 0
          delta = delta * (1. + decay);
        elseif alpha < rand()*.3 && zc == L
          delta = delta / (1. + decay);
        end
      end
    else
      if alpha > rand()*.7 + .3
        E_total = E_total * (1. + decay);
      elseif alpha < rand()*.3
        E_total = E_total / (1. + decay);
      end
    end
    decay = decay * decay_factor;
    rho = rho * decay_factor;
  end
  data(:,end+1) = [delta, E_total,Alpha];
  if i > 1
    energies(:,end+1) = [energies(1,end)*.9 + .1*U(xStar), energies(2,end)*.9 + .1*K(pStar),energies(3,end)*.9 + .1*(U(xStar)+K(pStar))];
  else
    energies(:,end+1) = [U(x(:,end)), K(p(:,end)),U(x(:,end))+K(p(:,end))];
  end
  ## LS(end+1) = L;
end
figure;
cov(x(:,500:end)')
x = x(:,BURNIN2:end);
plot(x(1,:),x(2,:),'.');

hold on;
t = linspace(0,2*pi,100)';
xy = 3*cov_hat_half*[cos(t)';sin(t)'];
plot(xy(2,:),xy(1,:),'-','Linewidth',2);
axis equal;
figure
plot(1:episode,data(3,:),'linewidth',2)
ylabel('\alpha')
xlabel('episode');
figure;
plot(1:episode,data(1,:),'linewidth',2)
ylabel('\delta');
xlabel('episode');
figure;
semilogy(1:episode,energies(1,:),1:episode,energies(2,:),1:episode,energies(3,:),'k-')
hold on;
legend('Potential energy','Kinetic energy', 'Total energy')
#print -dpng figures/hmc39_a5.png
#if 0
figure;
plot(dst_sigma(1,:),dst_sigma(2,:),'linewidth',2);
xlabel('episodes');
ylabel('|\Sigma_{est} \Sigma|');
figure;
plot(dst_mu(1,:),dst_mu(2,:),'linewidth',2);
xlabel('episodes');
ylabel('|\mu_{est} \mu|');
##                                 #x(:,end-100:end)
%%dst

                                #end
#print -dpng figures/hmc39_a6.png
#figure
#plot(1:length(LS),LS);
d = energies(3,:);
d1 = d(1:end-2);
d2 = d(2:end-1);
d3 = d(3:end);
d123 = (d1-2*d2+d3);
figure
plot(1:length(d123),d123);
pause();
