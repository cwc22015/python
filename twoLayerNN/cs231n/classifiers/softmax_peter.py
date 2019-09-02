    num_classes = W.shape[1]
    num_samples = X.shape[0]

    for i in range(num_samples):
        # Compute vector of scores
        f = X[i].dot(W)

        # Apply numerical stability term
        # Shift values so that highest value is 0.
        f -= np.max(f)

        # Compute loss (and add it up and divided later)
        sum_i = np.sum(np.exp(f))
        loss += -f[y[i]] + np.log(sum_i)

        # Compute gradient
        for j in range(num_classes):
            dW[:, j] += np.exp(f[j])/sum_i * X[i]
        # 'Fix' the weights for the true values.
        dW[:, y[i]] -= X[i]

    # Compute average
    loss /= num_samples
    dW /= num_samples

    # Regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    
    
# softmax_loss_vectorized example code (Peter):

    num_samples = X.shape[0]
    # Compute vector of scores
    f = X.dot(W)
    # Apply numerical stability term
    f -= np.max(f)

    sum_exp = np.exp(f).sum(axis=1, keepdims=True)
    softmax = np.exp(f)/sum_exp
    loss = np.sum(-np.log(softmax[np.arange(num_samples), y]) )

    # Weight Gradient
    softmax[np.arange(num_samples), y] -= 1
    dW = X.T.dot(softmax)

    # Compute average
    loss /= num_samples
    dW /= num_samples

    # Regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    
>>>>

	
