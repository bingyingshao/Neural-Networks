function [nabla_w,nabla_b] = backprop(x,y)
global s
global reshape
global weights
global bias
% create delta weights and bias matrices
nabla_w = {};
nabla_b = {};
for i = 1:length(reshape)
    nabla_w{i} = zeros(size(weights{i}));
    nabla_b{i} = zeros(size(bias{i}));
end
activation = x;
activations = {};
activations{1} = x;
zs = {};
% feedforward pass
for i = 1:length(reshape)
    z = bsxfun(@plus,(weights{i}*activation),bias{i});
    zs{i} = z;
    activation = sigmoid(z);
    activations{i+1} = activation; 
end
% backward pass
% 1. calcute "delta" (error_derivative) for output and target
% 2. calcute nabla_b = delta for the last hidden units
% 3. calcute nabla_w = delta times activations of the previous hidden units
delta = cost_error(activations{length(reshape)+1},y) .* sigmoid_prime(zs{length(reshape)});
nabla_b{length(reshape)} = delta;
nabla_w{length(reshape)} = delta * (activations{length(reshape)}');
% backprop the error to each hidden layer
for l = (length(s)-1) : -1 : 2
    z = zs{l-1};
    sp = sigmoid_prime(z);
    delta = ((weights{l}') * delta) .* sp;
    nabla_b{l-1} = delta;
    nabla_w{l-1} = delta * (activations{l-1}');
end
% sumup the nabla_b for all instances, nabla_w for all instances have already been summed up
for i = 1:length(reshape)
    nabla_b{i} = (sum(nabla_b{i},2));
    %weights{i} = ((nabla_w{i}) * eta) ./ m;
end
end
    
    
   
    
    