function p=prediction(x,w,l)
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
[dummy, p] = max(a{nl-1}, [], 1);
endfunction
