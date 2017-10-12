function [reshape,weights,bias] = ini(x,y,nodelayer)
global s
global reshape
global weights
global bias
%define reshape as the number of layers
s = nodelayer
reshape = s(2:end);
weights = {};
bias = {};
%initialize the weights and bias matrics
for i = 1:length(reshape)
    weights{i} = randn(s(i+1),s(i));
    bias{i} = randn(reshape(i),1);
end


