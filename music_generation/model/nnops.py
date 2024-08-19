import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import functools
import math
import keras

vars = {} # holds all the trainable variables in the network
settings = {'data_format': 'NHWC', 'weight_scale': True, 'spectral_normalization': True}

def get_trainable_variables() -> dict:
    trainable_variables = {}
    for name, var in vars.items():
        if 'param' not in name:
            trainable_variables[name] = var
    return trainable_variables

def load(dir: str):
    checkpoint_reader = tf.train.load_checkpoint(dir)
    var_to_shape_map = checkpoint_reader.get_variable_to_shape_map()
    for var_name in var_to_shape_map:
        name = var_name[:str.find(var_name, '/')]
        variable = tf.Variable(initial_value=checkpoint_reader.get_tensor(var_name), trainable=True, name=name)
        vars[name] = variable

def save(dir: str):
    checkpoint = tf.train.Checkpoint(**get_trainable_variables())
    checkpoint.save(dir+'model_checkpoint')


# the paper uses a random normal initialization and scales at runtime

def get_weight(name: str, variance=2.0):
    shape = vars[name].shape
    fan_in = np.prod(shape[1:])
    std = math.sqrt(variance / fan_in)
    if settings['weight_scale']:
        return vars[name] * std
    else:
        # this means the variable already uses xavier / he init
        return vars[name]

def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0, axis=-1, from_logits=False, class_balancing=False):
    if from_logits: 
        y_pred = tf.nn.softmax(y_pred, axis)
    epsilon = keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0-epsilon)
    p_t = y_true*y_pred+(1.0-y_true)*(1.0-y_pred)
    factor = (1.0 - p_t) ** gamma
    bce = -(y_true*tf.math.log(y_pred)+(1.0-y_true)*tf.math.log(1.0-y_pred))
    focal_bce = factor*bce
    if class_balancing:
        focal_bce *= y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
    return tf.reduce_sum(focal_bce, axis=axis, keepdims=False)

def bce(y_pred, y_true, axis=-1, from_logits=False):
    if from_logits: 
        y_pred = tf.nn.softmax(y_pred, axis)
    epsilon = keras.backend.epsilon()
    loss = -(y_true*tf.math.log(y_pred+epsilon) + (1.0-y_true)*tf.math.log(1.0-y_pred+epsilon))
    return tf.reduce_sum(loss, axis=axis, keepdims=False)

def create_batch_norm_vars(input, name: str, axis=-1):
    shape = input.shape.as_list()
    param_shape = [1] * len(shape)
    param_shape[axis] = shape[axis]
    vars[name+'_gamma'] = tf.Variable(tf.zeros(param_shape), name=name+"_gamma", trainable=True)
    vars[name+'_beta'] = tf.Variable(tf.ones(param_shape), name=name+"_beta", trainable=True)
    vars['param_'+name+'_moving_mean'] = tf.Variable(tf.zeros(param_shape), name=name+"_moving_mean", trainable=False)
    vars['param_'+name+'_moving_variance'] = tf.Variable(tf.ones(param_shape), name=name+"_moving_variance", trainable=False)

def create_dense_vars(input, units: int, name: str, weight_initializer=keras.initializers.RandomNormal(0.0, 1.0), bias_initializer=keras.initializers.Zeros(), variance=2.0):
    if not settings['weight_scale']:
        fan_in = np.prod(input.shape[1:])
        std = math.sqrt(variance / fan_in)
        weight_initializer.stddev = std

    shape = input.shape.as_list()
    vars[name+'_weight'] = tf.Variable(weight_initializer(shape=(shape[-1], units)), name=name+"_weight", trainable=True)
    vars[name+'_bias'] = tf.Variable(bias_initializer(shape=(1, units)), name=name+"_bias", trainable=True)

# just realized this is useless and probably wrong :skull:

def calc_same_padding(input_shape, kernel, stride, dilation, output_padding=(0, 0), transposed=False):
    hin = input_shape[-3]
    win = input_shape[-2]
    
    if transposed:
        oh = stride[0] * (hin - 1) + kernel[0] - 2 * output_padding[0]
        ow = stride[1] * (win - 1) + kernel[1] - 2 * output_padding[1]
        
        ph = kernel[0] - stride[0] + 2 * output_padding[0]
        pw = kernel[1] - stride[1] + 2 * output_padding[1]
        
        return (ph, pw)
    else:
        oh = int((hin + 2 * 0 - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1)
        ow = int((win + 2 * 0 - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1)
        
        kh = kernel[0] + (kernel[0] - 1) * (dilation[0] - 1)
        kw = kernel[1] + (kernel[1] - 1) * (dilation[1] - 1)
        
        ph = math.ceil((stride[0] * (oh - 1) + kh - hin) / 2)
        pw = math.ceil((stride[1] * (ow - 1) + kw - win) / 2)
        
        return (ph, pw)

# using YXIO weight format and NHWC here (bc we are on cpu :skull:)
def create_conv2d_vars(input, filters: int, name: str, kernel=(2, 2), stride=(1, 1), padding=(0, 0), dilation=(1, 1), weight_initializer=keras.initializers.RandomNormal(0.0, 1.0), bias_initializer=keras.initializers.Zeros(), variance=2.0):
    if not settings['weight_scale']:
        fan_in = np.prod(input.shape[1:])
        std = math.sqrt(variance / fan_in)
        weight_initializer.stddev = std
    shape = input.shape.as_list()
    vars[name+'_kernel'] = tf.Variable(weight_initializer(shape=(*kernel, shape[-1], filters)), name=name+"_kernel", trainable=True)
    # need to calculate output dims for the bias
    oh, ow = 0, 0
    if padding == 'SAME':
        oh = math.ceil(float(shape[-3])/float(stride[0]))
        ow = math.ceil(float(shape[-2])/float(stride[1]))
    else:
        oh = int((shape[-3] + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1)
        ow = int((shape[-2] + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1)
    vars[name+'_bias'] = tf.Variable(bias_initializer(shape=(1, oh, ow, filters)), name=name+"_bias", trainable=True)
    vars['param_'+name+'_stride'] = stride
    vars['param_'+name+'_padding'] = padding
    vars['param_'+name+'_dilation'] = dilation

def create_transposed_conv2d_vars(input, filters: int, name: str, kernel=(2, 2), stride=(1, 1), padding=(0, 0), dilation=(1, 1), weight_initializer=keras.initializers.RandomNormal(0.0, 1.0), bias_initializer=keras.initializers.Zeros(), variance=2.0):
    if not settings['weight_scale']:
        fan_in = np.prod(input.shape[1:])
        std = math.sqrt(variance / fan_in)
        weight_initializer.stddev = std
    shape = input.shape.as_list()
    # has to be output channels THEN input channels not like i carelessly missed before
    vars[name+'_kernel'] = tf.Variable(weight_initializer(shape=(*kernel, filters, shape[-1])), name=name+"_kernel", trainable=True)
    # need to calculate output dims for the bias
    oh, ow = 1, 1
    if padding == 'SAME':
        oh = shape[-3]*stride[0]
        ow = shape[-2]*stride[1]
    else:
        oh = stride[0]*(shape[-3]-1)+kernel[0]+(kernel[0]-1)*(dilation[0]-1)-2*padding[0]
        ow = stride[1]*(shape[-2]-1)+kernel[1]+(kernel[1]-1)*(dilation[1]-1)-2*padding[1]
    vars[name+'_bias'] = tf.Variable(bias_initializer(shape=(1, oh, ow, filters)), name=name+"_bias", trainable=True)
    vars['param_'+name+'_output_shape'] = [shape[0], oh, ow, filters]
    vars['param_'+name+'_stride'] = stride
    vars['param_'+name+'_padding'] = padding
    vars['param_'+name+'_dilation'] = dilation

@tf.function
def batch_norm(input: tf.Tensor, training: bool, name: str, momentum=0.99, epsilon=1.0e-7, axis=-1):
    shape = input.shape.as_list()
    reduction_axes = list(range(len(shape)))
    reduction_axes.pop(axis)

    gamma = get_weight(name+'_gamma')
    beta = get_weight(name+'_beta')
    moving_mean = get_weight('param_'+name+'_moving_mean')
    moving_variance = get_weight('param_'+name+'_moving_variance')

    mean, variance = tf.nn.moments(x=input, axes=reduction_axes, keepdims=True)
    # works because we use @tf.function, this gets converted to tf.cond anyway
    if training:
        moving_mean = moving_mean*momentum+mean*(1.0-momentum)
        moving_variance = moving_variance*momentum+variance*(1.0-momentum)
    else:
        mean = moving_mean
        variance = moving_variance

    stddev = tf.sqrt(variance + epsilon)
    input = (input - mean) / stddev
    input = input * gamma + beta
    return input

def pixel_normalization(input, axis=1, epsilon=1.0e-12):
    pixel_norm = tf.sqrt(tf.reduce_mean(tf.square(input), axis=axis, keepdims=True) + epsilon)
    return input / pixel_norm

# repeat along axes 1 and 2 (nearest neighbor upsampling)
def upscale2d(input, factors=[2, 2]):
    if type(factors) is int:
        factors = [factors, factors]
    input = tf.repeat(input, repeats=factors[0], axis=1)
    input = tf.repeat(input, repeats=factors[1], axis=2)
    return input

def downscale2d(inputs, factors=[2, 2]):
    # because we r using numpy
    if (factors == 1).all():
        return inputs
    return tf.nn.avg_pool(
        input=inputs,
        ksize=[1, *factors, 1],
        strides=[1, *factors, 1],
        padding='SAME'
    )


def max_pooling2d(input, kernel_size, strides):
    return tf.nn.max_pool(
        value=input,
        ksize=[1, *kernel_size, 1],
        strides=[1, *strides, 1],
        padding="SAME",
    )

def avg_pooling2d(input, kernel_size, strides):
    return tf.nn.avg_pool(
        value=input,
        ksize=[1, *kernel_size, 1],
        strides=[1, *strides, 1],
        padding="SAME",
    )

def flatten(input):
    shape = input.shape.as_list()
    flattened_dim = 1
    for i in range(1, len(shape)):
        flattened_dim *= shape[i]
    return tf.reshape(input, [-1, flattened_dim])

def dense(input, units, name, variance=2.0):
    if name not in vars:
        create_dense_vars(input, units, name)
    return tf.matmul(input, get_weight(name+'_weight', variance)) + vars[name+'_bias']

def conv2d(input, filters, name, kernel=(2, 2), stride=(1, 1), padding=(0, 0), dilation=(1, 1), variance=2.0):
    if name not in vars:
        create_conv2d_vars(input, filters, name, kernel, stride, padding, dilation)
    return tf.nn.conv2d(input=input, filters=get_weight(name+'_kernel', variance), strides=vars['param_'+name+'_stride'], padding=vars['param_'+name+'_padding'], dilations=vars['param_'+name+'_dilation']) + vars[name+'_bias']

def conv2d_transpose(input, filters, name, kernel=(2, 2), stride=(1, 1), padding=(0, 0), dilation=(1, 1), variance=2.0):
    if name not in vars:
        create_transposed_conv2d_vars(input, filters, name, kernel, stride, padding, dilation)
    return tf.nn.conv2d_transpose(input=input, filters=get_weight(name+'_kernel', variance), output_shape=vars['param_'+name+'_output_shape'], strides=vars['param_'+name+'_stride'], padding=vars['param_'+name+'_padding'], dilations=vars['param_'+name+'_dilation']) + vars[name+'_bias']

def log(x, base):
    return tf.math.log(x) / tf.math.log(base)

def lerp(a, b, t):
    return a + (b - a) * t

# unused

def embedding(input, units):
    shape = input.shape.as_list()
    if 'embedding_'+str(units) not in vars:
        vars['embedding_'+str(units)] = tf.Variable(keras.initializers.RandomNormal()(shape=(shape[-1], units)), trainable=True, name='embedding_'+str(units))
    return tf.nn.embedding_lookup(get_weight('embedding_'+str(units)), tf.argmax(input, axis=-1))

def interpolate(points: list[tuple], xi: int, n: int) -> float:
    result = 0.0
    for i in range(n):
        term = points[i][1]
        for j in range(n):
            if j != i:
                term = term * (xi - points[j][0]) / (points[i][0] - points[j][0])
        result += term
    return result
