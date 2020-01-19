%% newest workable code
rand('seed',54321);
randn('seed',54321);

%% parameter for HMC
D = 2;
k1=99.;
SIGMA=[1,k1/100.;k1/100.,1];
%%SIGMA=[k1,1/k1;1/k1,k1];
ISIGMA = inv(SIGMA);
U = @(x) x' * ISIGMA * x/2;
dU = @(x) ISIGMA *x;
x0 = zeros(D,1);
if 1
[x, p] = ghmc(U,dU, D, x0, EPISODE=10000);
else
  [x, p] = hmc(U,dU, D, x0, EPISODE=10000,delta=.5);
end

cov(x(:,2000:end)')
figure;
plot(x(1,2000:end),x(2,2000:end),'.');
axis equal;
hold on;
t = linspace(0,2*pi,100)';
xy = 3*sqrtm(SIGMA)*[cos(t)';sin(t)'];
plot(xy(2,:),xy(1,:),'-','Linewidth',2);

pause;
