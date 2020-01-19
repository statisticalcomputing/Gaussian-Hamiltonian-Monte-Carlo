%% demonstrate the effect of spatial transformation
#rand('seed',54321);
#randn('seed',54321);
rLim = 5;
BND = rLim + 1;
%% STEP SIZE
delta = .05;
L = 10;
episode = 1000;
SIGMA=[1,.99;.99,1];
#SIGMA=[99,1/99;1/99,99];
%% DEFINE POTENTIAL ENERGY FUNCTION
x0 =[0;0];
U = @(x) (x-x0)' * inv(SIGMA) * (x-x0)/2;
%% DEFINE GRADIENT OF POTENTIAL ENERGY
dU = @(x) inv(SIGMA)*(x-x0);

%% DEFINE GRADIENT OF POTENTIAL ENERGY
K = @(x) x' *  x/2;
dK = @(x) x;

for i=1:3
  x = [];
  z = [];
  p = [];
  zStar = randn(2,1)*2;
  pStar = randn(2,1)*2;
  z2x = @(z) sqrtm(SIGMA) * z + x0;
  x(:,end+1) = [z2x(zStar);U(z2x(zStar))];
  p(:,end+1) = [pStar;K(pStar)];
  z(:,end+1) = [zStar;U(z2x(zStar))];
  for j= 1:episode
    zStar = zStar + delta*dK(pStar);
    pStar = pStar - delta*sqrtm(SIGMA)*dU(z2x(zStar));
    x(:,end+1) = [z2x(zStar);U(z2x(zStar))];
    z(:,end+1) = [zStar;U(z2x(zStar))];
    ##z(:,end+1) = [zStar;U(z2x(zStar))];
    p(:,end+1) = [pStar;K(pStar)];
  end
  #figure(1);
  subplot(1,3,3);
  plot3(x(1,500:end),x(2,500:end),x(3,500:end),'r-','Linewidth',1);
  hold on;
  plot(x(1,500:end),x(2,500:end),'b-','Linewidth',2);

  subplot(1,3,1);
  plot3(z(1,500:end),z(2,500:end),z(3,500:end),'r-','Linewidth',1);
  hold on;
  plot(z(1,500:end),z(2,500:end),'b-','Linewidth',2);
  #plot3(x(1,1),x(2,1),'bo','Markersize',20);

  #figure(2);
  subplot(1,3,2);
  plot3(p(1,500:end),p(2,500:end),p(3,500:end),'r-','Linewidth',1);
  hold on;
  plot(p(1,500:end),p(2,500:end),'b-','Linewidth',2);
  
  
  #legend('momentum-kinetic traj.','spatial traj.','3 \sigma boundary')
  #                               #print -dpng 'figures/hmc2-1.png'
end
#figure(1)
subplot(1,3,3);
hold on;
t = linspace(0,2*pi,100)';
xy = 3*sqrtm(SIGMA)*[cos(t)';sin(t)'] + x0;
%% ## plot(circsx,circsy,'k');
plot3(xy(2,:),xy(1,:),'k','Linewidth',3);
xlabel('x_1');
ylabel('x_2');
zlabel('potential');
title('spatial')
legend('spatial-potential','spatial traj.')
                                #print -dpng 'figures/hmc2-1.png'

subplot(1,3,1);
hold on;
t = linspace(0,2*pi,100)';
xy = 3*[cos(t)';sin(t)'] + x0;
%% ## plot(circsx,circsy,'k');
plot3(xy(2,:),xy(1,:),'k','Linewidth',3);
xlabel('z_1');
ylabel('z_2');
zlabel('potential');
title('transformed spatial')
legend('transformed spatial-potential','trans. spatial traj.')
                                #print -dpng 'figures/hmc2-1.png'
axis equal
subplot(1,3,2);
hold on;
#figure(2);
t = linspace(0,2*pi,100)';
                                #xy = 3*[cos(t)';sin(t)'];
xy = 3*[cos(t)';sin(t)'];
%% ## plot(circsx,circsy,'k');
plot3(xy(2,:),xy(1,:),'k','Linewidth',3);

##plot3(x(1,1),x(2,1),'bo','Markersize',20);
xlabel('p_1');
ylabel('p_2');
zlabel('kinetic');
title('momentum')
legend('momentum-kinetic','momentum traj.')
                                #axis equal;
axis equal
pause()
