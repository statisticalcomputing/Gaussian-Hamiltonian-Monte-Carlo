%% show the Riemann HMC approach, i.e. momentum transformation
%%rand('seed',54321);
%%randn('seed',54321);
rLim = 5;
BND = rLim + 1;
%% STEP SIZE
delta = .05;
L = 10;
episode = 1000;
SIGMA=[1,.9;.9,1];
#SIGMA=[1.0,0.0;
#       0.0,1.0];
%% DEFINE POTENTIAL ENERGY FUNCTION
U = @(x) x' * inv(SIGMA) * x/2;
%% DEFINE GRADIENT OF POTENTIAL ENERGY
dU = @(x) inv(SIGMA)*x;

%% DEFINE KINETIC ENERGY FUNCTION
#K = @(p) (transpose(p)*p)/2;
#dK = @(p) p;
K = @(x) x' * SIGMA* x/2;
%% DEFINE GRADIENT OF POTENTIAL ENERGY
dK = @(x) SIGMA*x;
for i=1:3
%% INITIAL STATE
x = [];
xStar = randn(2,1);
pStar = randn(2,1);
x(:,end+1) = [xStar;U(xStar)];
p(:,end+1) = [pStar;K(pStar)];
%% SAMPLE RANDOM MOMENTUM

for j= 1:episode
  xStar = xStar + delta*dK(pStar);
  pStar = pStar - delta*dU(xStar);
  x(:,end+1) = [xStar;U(xStar)];
  p(:,end+1) = [pStar;K(pStar)];
end
                                #subplot(1,2,1)
#figure(1);
subplot(1,2,1);
plot3(x(1,:),x(2,:),log(x(3,:)),'r-','Linewidth',1);
hold on;
plot(x(1,:),x(2,:),'b-','Linewidth',2);
#plot3(x(1,1),x(2,1),'bo','Markersize',20);

                                #print -dpng 'figures/hmc2-1.png'
#figure(2);
subplot(1,2,2);
#subplot(1,2,2)
plot3(p(1,:),p(2,:),p(3,:),'r-','Linewidth',1);
hold on;
plot(p(1,:),p(2,:),'b-','Linewidth',2);

                                #print -dpng 'figures/hmc2-1.png'
end
#figure(1)
subplot(1,2,1);

hold on;
t = linspace(0,2*pi,100)';
xy = 3*sqrtm(SIGMA)*[cos(t)';sin(t)'];
plot3(xy(2,:),xy(1,:),'k.','Linewidth',3);

xlabel('x_1');
ylabel('x_2');
zlabel('log potential');
title('spatial')
legend('spatial-potential','spatial traj.')
axis equal
subplot(1,2,2);
hold on;
t = linspace(0,2*pi,100)';
xy = 3*sqrtm(inv(SIGMA))*[cos(t)';sin(t)'];
xy1 = 3*[cos(t)';sin(t)'];
%% ## plot(circsx,circsy,'k');
#plot3(xy(2,:),xy(1,:),'m','Linewidth',3);
#plot3(xy1(2,:),xy1(1,:),'k','Linewidth',3);
##plot3(x(1,1),x(2,1),'bo','Markersize',20);
xlabel('p_1');
ylabel('p_2');
zlabel('kinetic');
#zlabel('kinetic');
title('momentum')
legend('momentum-kinetic','momentum traj.');
axis equal
pause()
