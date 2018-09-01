import tensorflow as tf

def LeNet2(x, keep_prob=1.):
    with tf.name_scope('Model'):
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        mu = 0
        sigma = 0.1
        
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
        conv1_b = tf.Variable(tf.zeros(6))
        conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        print("conv1 shape:",conv1.get_shape())
        
        # Activation.
        conv1 = tf.nn.elu(conv1)

        # Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Layer 2: Convolutional. Output = 10x10x16.
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        
        # Activation.
        conv2 = tf.nn.elu(conv2)

        # Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        ppp = conv2
        
        # Layer 3: Convolutional. Output = 1x1x400.
        conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 400), mean = mu, stddev = sigma))
        conv3_b = tf.Variable(tf.zeros(400))
        conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
        
        # Activation.
        conv3 = tf.nn.elu(conv3)
        
        # Flatten. Input = 5x5x16. Output = 400.
        fc0   = tf.contrib.layers.flatten(ppp)

        # Flatten. Input = 5x5x16. Output = 400.
        fc1   = tf.contrib.layers.flatten(conv3)
        
        # Concat layer2flat and x. Input = 400 + 400. Output = 800
        concat_x = tf.concat([fc1, fc0], 1)
        
        # Dropout
        concat_x = tf.nn.dropout(concat_x, keep_prob)
        
        # Layer 4: Fully Connected. Input = 800. Output = 43.
        fc2_W = tf.Variable(tf.truncated_normal(shape=(800, 43), mean = mu, stddev = sigma))
        fc2_b = tf.Variable(tf.zeros(43))
        logits = tf.matmul(concat_x, fc2_W) + fc2_b
    
    return logits
    