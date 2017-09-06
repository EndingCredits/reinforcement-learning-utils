import numpy as np
import tensorflow as tf
import ops
from layers import linear, conv2d, set_layer
from ops import flatten, mask_and_pool


def deepmind_CNN(state, output_size=128):
    initializer = tf.truncated_normal_initializer(0, 0.1)
    activation_fn = tf.nn.relu
    
    state = tf.transpose(state, [0, 2, 3, 1])
    
    l1 = conv2d(state, 32, [8, 8], [4, 4], initializer, activation_fn, 'NHWC', name='l1')
    l2 = conv2d(l1, 64, [4, 4], [2, 2], initializer, activation_fn, 'NHWC', name='l2')
    l3 = conv2d(l2, 64, [3, 3], [1, 1], initializer, activation_fn, 'NHWC', name='l3')

    shape = l3.get_shape().as_list()
    l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
      
    embedding = linear(l3_flat, output_size, activation_fn=activation_fn, name='l4')

    # Returns the network output, parameters
    return embedding


def feedforward_network(state, out_size=128):
    initializer = tf.truncated_normal_initializer(0, 0.1)
    activation_fn = tf.nn.relu

    l1 = linear(state, 64,
      activation_fn=activation_fn, name='l1')
    l2 = linear(state, 64,
      activation_fn=activation_fn, name='l2')

    embedding = linear(l2, out_size,
      activation_fn=activation_fn, name='l3')

    # Returns the network output, parameters
    return embedding


def rav_layer(x, mask, out_size, **kwargs):
    x = x - mask_and_pool(x, mask)
    out = set_layer(x, **kwargs)
    return out
    

def embedding_network(state, mask):
    # Placeholder layer sizes
    d_e = [[128, 256]]
    d_o = [128]

    # Build graph:
    initial_elems = state
    
    # Get mask
    mask = ops.get_mask(state)

    # Embedding Part
    for i, block in enumerate(d_e):
        el = initial_elems
        for j, layer in enumerate(block):
            context = c if j==0 and not i==0 else None
            el = set_layer(el, layer, context=context, name='l'+str(i)+'_'+str(j))

        c = mask_and_pool(el, mask) # pool to get context for next block
    
    # Output
    embedding = c

    return embedding
    
    
def object_embedding_network(state, n_actions=128):
    mask = ops.get_mask(state)
    net = embedding_network(state, mask)
    out = ops.linear(net, n_actions, name='outs')
    return out

