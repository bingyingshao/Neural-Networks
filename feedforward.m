function [a] = feedforward(a)
global reshape
global weights
global bias
% Lastly, run feedforward one time for getting final outputs
for i = 1:length(reshape)
    z = bsxfun(@plus,(weights{i}*a),bias{i});
    a = sigmoid(z);
end
end
    