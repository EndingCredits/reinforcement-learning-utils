import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

def get_mask(x):
    # Returns a matrix with values set to 1 where elements aren't padding
    emb_sum = tf.reduce_sum(tf.abs(x), axis=-1, keep_dims=True)
    return 0.99 - tf.to_float(tf.equal(emb_sum, 0.0))

def mask_and_pool(embeds, mask, pool_type='max'):
    # Use broadcasting to multiply
    masked_embeds = tf.multiply(embeds, mask)

    # Pool using max pooling
    embed = tf.reduce_max(masked_embeds, axis=1)
    return embed

def flatten(input_):
    in_list = [x for x in input_ if x is not None]
    if type(in_list[0]) is list:
      in_list = [flatten(elem) for elem in in_list ]
    
    return tf.concat([ tf.reshape(elem, [-1]) for elem in in_list], axis=0)
    
# Gets the indexes specified by the lengths, equivalent to doing output[:, lengths, :]
def last_relevant(output, lengths):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (lengths - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant
    
def batch_lists(input_list):
    # Takes an input list of lists (of vectors), pads each list the length of
    # the longest list, compiles the list into a single n x m x d array, and 
    # returns a corresponding n x m x 1 mask.
    max_len = 0
    out = []; masks = []
    for i in input_list: max_len = max(len(i),max_len)
    for l in input_list:
        # Zero pad output
        out.append(np.pad(np.array(l, dtype=np.float32), ((0,max_len-len(l)),(0,0)), mode='constant'))
        # Create mask...
        masks.append(np.pad(np.array(np.ones((len(l),1)),dtype=np.float32), ((0,max_len-len(l)),(0,0)), mode='constant'))
    return out, masks


