import numpy as np 
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# optional packages
try:
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.WARN)
except:
    logging.info('Import Error.\nCould not import tensorflow. Install tensorflow to use nnet derivatives.\n')

####################################################################################################################################################
# Helper functions
####################################################################################################################################################

def __minibatch_feeder__(U, X, minibatch_size = 100, subsample = 1):
    """
    Function for passing examples pairs of data x,u to stochastic optimization algorithm

    Input:
        U               : n-D numpy array of data
        X               : data spacing for all dimensions.  i.e. can be [dx, dy, dz, dt] for 4D cube
        minibatch_size  : size to pass back at each call
        subsample       : fraction of the data to subsample
    """

    # What size is the data?
    dims = U.shape
    N = np.product(dims)
    d = len(dims)

    if len(X) != d: raise Exception('len(X) != len(U.shape)')

    # Enumerate locations
    X_flat = np.hstack([np.tile(np.array([j*np.ones(int(np.product(dims[:i]))) \
        for j in X[i]]).flatten(), \
        int(np.product(dims[i+1:]))).reshape(N,1) for i in range(d)])
    Ur = U.reshape(N,1, order = 'F')

    # Subsample random points in space and time
    if subsample != 1:

        N_sample = int(N*subsample)
        sample_points = np.random.choice(N,N_sample, replace= False)

        Ur = Ur[sample_points].reshape(N_sample, 1)
        X_flat = X_flat[sample_points,:].reshape(N_sample, d)

    else: N_sample = N


    # Initialize a queue
    queue = list(np.random.permutation(N_sample))

    while True:
        
        # Ensure sufficient queue length
        if len(queue) < minibatch_size:
            queue = queue + list(np.random.permutation(N_sample))
            
        # Take top entries from queue
        minibatch, queue = queue[:minibatch_size], queue[minibatch_size:]
        
        # Yield minibatch
        yield (X_flat[minibatch].T, Ur[minibatch].T)


def __prepare_network__(x, t, m, layer_sizes):

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [1,None]) # will be (x,t)
    Y = tf.placeholder(tf.float32, [1,None]) # u(x,t)

    def dense_layer(x, W, b, last = False):
        x = tf.matmul(W,x)
        x = tf.add(x,b)
                   
        if last: return x
        else: return tf.sin(x)

    def simple_net(x, weights, biases):
        
        layers = [x]
        
        for l in range(len(weights)-1):
            layers.append(dense_layer(layers[l], weights[l], biases[l]))

        out = dense_layer(layers[-1], weights[-1], biases[-1], last = True)
        
        return out

    
    num_layers = len(layer_sizes)

    weights = []
    biases = []

    for j in range(1,num_layers):
        weights.append(tf.get_variable("W"+str(j), [layer_sizes[j],layer_sizes[j-1]], \
                                       initializer = tf.contrib.layers.xavier_initializer(seed = 1)))
        biases.append(tf.get_variable("b"+str(j), [layer_sizes[j],1], initializer = tf.zeros_initializer()))
        
    # Construct model
    prediction = simple_net(X, weights, biases)

    # Define cost function and optimizer
    lr = tf.placeholder(tf.float32)    # learning rate
    beta = tf.placeholder(tf.float32)  # L2 regularization parameter

    cost = tf.losses.mean_squared_error(Y,prediction)
    regularizer = tf.reduce_sum([tf.nn.l2_loss(W) for W in weights])
    cost = tf.reduce_sum(cost + beta * regularizer)

    
    return X, Y, prediction, cost, lr, beta

####################################################################################################################################################
# Adam optimizer
####################################################################################################################################################

def adam(x, dt, params, options={'layer_sizes': [1,50,50,50,1], 'beta': 1e-5}):
    '''
    Use a neural network to learn the dynamics in order to take the derivative. Using the Adam optimizer.
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list)  [num_epochs,          : (int) number of epochs (e.g. 300)
                       minibatch_size,      : (int) size of each batch (e.g. 10)
                       base_learning_rate,  : (float) learning rate for first epoch (e.g. 0.001)
                       learning_rate_decay] : (float) time constant: learning_rate = base_learning_rate*exp(-epoch/learning_rate_decay. (e.g. 200)

    options : (dict) {'layer_sizes': [1,50,50,50,1],  : (list, required)  list of layer sizes. 
                      'beta': 1e-5}                   : (float, required) regularization parameter, favors smaller weights in connectivity matrices of the network       

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''
    
    num_epochs, minibatch_size, base_learning_rate, learning_rate_decay = params 
    layer_sizes = options['layer_sizes']
    beta_value = options['beta']

    t = np.arange(0, len(x)*dt, dt).reshape(1, -1)
    m = len(x)


    X, Y, prediction, cost, lr, beta = __prepare_network__(x, t, m, layer_sizes)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


    # Optimization parameters
    batches_per_epoch = int(m/minibatch_size) 
    # Note that implies if subsample<1, we're really passing over the data num_epochs/subsample times
    # This way the number of optimization steps is independent of the subsample rate

    # What (random) fraction of the noisy datapoints does the NN get to train on?
    subsample = 1

    # Set up generator
    MF = __minibatch_feeder__(x,[t],minibatch_size,subsample=subsample)

    # Do the training loop
    for epoch in range(1,num_epochs+1):

        epoch_cost = 0.
        learning_rate = base_learning_rate*np.exp(-epoch/learning_rate_decay)

        for batch in range(batches_per_epoch):

            # Select a minibatch
            (minibatch_X, minibatch_Y) = next(MF)

            # Run single step of optimizer
            _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, \
                                                                        lr:learning_rate, beta: beta_value})
            
            # Record cost
            epoch_cost += minibatch_cost / batches_per_epoch

        # Print the cost every epoch
        if 0:
            if epoch == 1 or epoch % 25 == 0: print ("Cost after epoch %i: %f" % (epoch, epoch_cost))    
        
    u_x_predictor = tf.gradients(prediction,X)[0]
    #u_xx_predictor = tf.gradients(u_x_predictor,X)[0]
    #u_xxx_predictor = tf.gradients(u_xx_predictor,X)[0]

    x_hat = prediction.eval({X: t}).T
    dxdt_hat = u_x_predictor.eval({X: t}).T

    return np.ravel(x_hat), np.ravel(dxdt_hat)

####################################################################################################################################################
# Quasi newton optimizer
####################################################################################################################################################

def quasinewton(x, dt, params, options={'layer_sizes': [1,50,50,50,1], 'beta': 1e-5}):
    '''
    Use a neural network to learn the dynamics in order to take the derivative. Using the Adam optimizer.
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (empty list)  []

    options : (dict) {'layer_sizes': [1,50,50,50,1],  : (list, required)  list of layer sizes. 
                      'beta': 1e-5}                   : (float, required) regularization parameter, favors smaller weights in connectivity matrices of the network 

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''
    layer_sizes = options['layer_sizes']
    beta_value = options['beta']

    t = np.arange(0, len(x)*dt, dt).reshape(1, -1)
    m = len(x)

    X, Y, prediction, cost, lr, beta = __prepare_network__(x, t, m, layer_sizes)

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(cost)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    optimizer.minimize(sess, feed_dict={X: t.reshape(1,m), Y: x.reshape(1,m), beta: beta_value}) 

    u_x_predictor = tf.gradients(prediction,X)[0]
    #u_xx_predictor = tf.gradients(u_x_predictor,X)[0]
    #u_xxx_predictor = tf.gradients(u_xx_predictor,X)[0]

    x_hat = prediction.eval({X: t}).T
    dxdt_hat = u_x_predictor.eval({X: t}).T

    return np.ravel(x_hat), np.ravel(dxdt_hat)

