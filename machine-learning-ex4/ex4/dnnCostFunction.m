function [J grad] = dnnCostFunction(w,x,y,l)
w1=w;
nl=length(l);
m=size(x,2);
%reshape function
for i=1:nl-1 
  r=l(i+1);
  c=l(i);
  weight{i}=reshape(w1(1:r*c),r,c);
  w1=w1(r*c+1:end);
  bias{i}=reshape(w1(1:r),r,1);
  w1=w1(r+1:end);
  size(weight{i});
  size(bias{i});
endfor

%forward propagation  
z{1}=weight{1}*x+bias{1};
a{1}=sigmoid(z{1});
for i=2:nl-1;
  z{i}=weight{i}*a{i-1}+bias{i};
  a{i}=sigmoid(z{i});
endfor

for i=1:nl-1;
  a{i};
endfor


y1=a{nl-1};
size(y1);
t=y.*log(y1)+(1-y).*log(1-y1);
size(t);
J=sum(sum((-1/m).*t));


%Back propagation
dz{nl-1}=y1-y;
dw{nl-1}=dz{nl-1}*a{nl-2}'/m;
db{nl-1}=sum(dz{nl-1},2)/m;

for i=nl-2:2
  dz{i}=weight{i+1}'*dz{i+1}.*(a{i}).*(1-a{i});
  dw{i}=dz{i}*a{i-1}'/m;
  db{i}=sum(dz{i},2)/m;  
endfor

dz{1}=weight{2}'*dz{2}.*(a{1}).*(1-a{1});
dw{1}=dz{1}*x'/m;
db{1}=sum(dz{1},2)/m;

grad=[];
for i=1:nl-1
  grad=[grad  dw{i}(:)' db{i}(:)'];
endfor
endfunction
