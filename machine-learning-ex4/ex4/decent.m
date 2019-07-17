function [J,w] = decent(costFunction,w,iter)
 alpha=1; 
for i=1:iter    
  [J grad]=costFunction(w);
  w=w-alpha*grad;
endfor
  
endfunction
