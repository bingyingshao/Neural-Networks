function [weights,bias] = update_mini_batch(x,y,eta)
global s
global reshape
global weights
global bias
% #of instance within one mini batch 
m = size(x,2);
% backprop algorithm
[nabla_w,nabla_b] = backprop(x,y);
% update weights and bias for each mini batch
for j = 1:length(reshape) 
    weights{j} = weights{j} - (eta/m).* (nabla_w{j}); 
    bias{j} = bias{j} - (eta/m).* (nabla_b{j});
end
end
