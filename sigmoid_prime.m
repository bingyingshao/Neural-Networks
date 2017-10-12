function output = sigmoid_prime(q)
output = sigmoid(q).* bsxfun(@minus,1,sigmoid(q));
end