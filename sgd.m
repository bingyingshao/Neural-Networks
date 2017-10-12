function  output = sgd(x,y,nodelayer,numEpochs,batchSize,eta)
%x and y have already been transposed before input
global s
global reshape
global weights
global bias
%run for each Epoch
for i = 1:numEpochs
    %random permutation for transposed x and y
    rand_id = randperm(size(x,2));
    x = x(:,rand_id);
    y = y(:,rand_id);
    %make batch round index list
    batch_id_lst = (0:floor(size(x,2)/batchSize));
    batch_n = length(batch_id_lst)-1;
    %create batch holders
    x_batches = {};
    y_batches = {};
    %add mini batches into holders  
    for i = 1:batch_n
        if i == 1
                x_batches{i}=x(:,(1:batch_id_lst(i+1)*batchSize));
                y_batches{i}=y(:,(1:batch_id_lst(i+1)*batchSize));
        else
                x_batches{i}=x(:,(batch_id_lst(i)*batchSize+1:batch_id_lst(i+1)*batchSize));
                y_batches{i}=y(:,(batch_id_lst(i)*batchSize+1:batch_id_lst(i+1)*batchSize));
        end
    end
    %run mini batch and update weights and bias
    for i = 1:batch_n
        x_mini_batch = x_batches{i};
        y_mini_batch = y_batches{i};
        [weights,bias] = update_mini_batch(x_mini_batch,y_mini_batch,eta);
    end
    %calculate mse and accuracy after one Epoch
    [mse,correct,acc] = evaluate(x,y);
    %stop when accuracy = 100%
    if acc == 1         
        fprintf('Parameters||Epoch %d, MSE: %f, Correct: %d/%d, Acc: %f \n',i, mse, correct, length(y), acc);
        return   
    end    
    %print result for each Epoch
    fprintf('Parameters||Epoch %d, MSE: %f, Correct: %d/%d, Acc: %f \n',i, mse, correct, length(y), acc);
end   
end

