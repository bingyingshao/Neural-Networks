function [mse,correct,acc] = evaluate(input,target)
% calculate mse
a = feedforward(input);
mse = mean(sum((a - target).^2,2)/length(target));
% calculate accuracy
% for XOR problem, output only has one node
if size(target,1) == 1
    y_pred = round(a);
    y_true = target;
% for other unbinary class, use argmax function on output nodes
else
    [argvalue_x, argmax_x] = max(a);
    [argvalue_y, argmax_y] = max(target);
    y_pred = argmax_x;
    y_true = argmax_y;
end
correct = 0;
for i = 1:size(input,2)
    if y_pred(i) == y_true(i)
        correct = correct + 1;
    end
end
acc = correct/length(target);
end




