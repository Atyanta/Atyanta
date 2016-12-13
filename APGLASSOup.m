function [c] = APGLASSOup(b,A,para)

%function [c,X] = APGLASSOup(b,A,para)

%% Object: c = argmin (1/2)\|y-Dx\|_2^2+lambda\|x\|_1+\mu\|x_I\|_2^2 
%%                s.t. x_T >0 
%% input arguments:
%%         b -------  D'*y transformed object vector
%%         A -------  D'*D transformed dictionary
%%         para ------ Lambda: Sparsity Level
%%                     (lambda1: template set; lambda2:trivial template; lambda3:\mu)
%%                     Lip: Lipschitz Constant for F(x)
%%                     Maxit: Maximal Iteration number
%%                     nT: number of templates
%% output arguments:
%%         c ------  output Coefficient vetor

%  Initialization
ColDim = size(A,1);
xPrev = zeros(ColDim,1);
x = zeros(ColDim,1);

tPrev = 1;
t = 1;
lambda = para.Lambda;
Lip = para.Lip;
%maxit = para.Maxit;
maxit = 20;

%X = zeros(ColDim,maxit);

nT = para.nT;

temp_lambda = zeros(ColDim,1);
temp_lambda(1:nT) = lambda(1);
temp_lambda(end) = lambda(1);  % fixT template

%warna = ['b' 'g' 'r' 'c' 'm' 'b' 'g' 'r' 'c' 'm'];

%% main loop
for iter =1:maxit
    tem_t = (tPrev-1)/t; %Mt_t(iter) = tem_t;
    tem_y = (1+tem_t)*x - tem_t*xPrev; %NormY(iter) = norm(tem_y);
    
    %Simulasi nilai dari tem_t dan tem_y
    %plot(tem_y),title(['Nilai dari tem_t: ',num2str(tem_t)]);
    %pause;
    
    temp_lambda(nT+1:end-1) = lambda(3)*tem_y(nT+1:end-1);%tem_tengah(iter)=norm(temp_lambda(nT+1:end-1));
    tem_y = tem_y - (A*tem_y-b+temp_lambda)/Lip; %NormY2(iter) = norm(tem_y);% update residual 
    xPrev = x;
    x(1:nT) = max(tem_y(1:nT),0);
    x(end) = max(tem_y(end),0);
    x(nT+1:end-1) = softthres(tem_y(nT+1:end-1),lambda(2)/Lip); %NormX(iter)=norm(x);
    tPrev = t;
    t = (1+sqrt(1+4*t^2))/2;
    %plot(x,warna(iter));
    %hold on
    %X(:,iter)=x;
end
%figure, plot(x,'m')
c = x;
% subplot(4,1,1),plot(Mt_t);title('ini adalah nilai dari tem_t');
% subplot(4,1,2),plot(NormY);title('ini adalah nilai dari Norm tem_y');
% subplot(4,1,3),plot(NormY2);title('ini adalah nilai akhir dari tem_Y');
% subplot(4,1,4),plot(NormX);title('ini adalah nilai akhir dari X');

%% soft thresholding operator
function y = softthres(x,lambda)
y = max(x-lambda,0)-max(-x-lambda,0);


    