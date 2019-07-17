

%load('ex4data1.mat');
%x=X';
%a=[1 2 3 4 5 6 7 8 9 10];
%y=(y==a);
%y=y';
%wi=rand(1,17070)';
%l=[400 40 20 10];
x=[1 12 2; 1 12 2];
y=[1 0 1;0 1 0];
t=[1 2 1];
l=[2 3 3 2];
wi=rand(29,1);
%options = optimset('MaxIter', 50);
costFunction = @(w)dnnCostFunction(w,x,y,l);
                                   
[j,wf] = decent(costFunction, wi,5000);

%wf=fminsearch(costFunction,w)


p=prediction(x,wf,l)

fprintf('\nTraining Set Accuracy: %f\n', mean(double(p == t)) * 100);