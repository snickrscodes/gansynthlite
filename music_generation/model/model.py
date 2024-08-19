import tensorflow as tf
import datetime
from scipy.io import wavfile
import numpy as np
import os
import nnops as ops
import spectral_ops as spec_ops
import math

spectral_params=dict(
            waveform_length=64000, 
            sample_rate=16000, 
            spectrogram_shape=[128, 1024], 
            overlap=0.75)

hyper_params=dict(
            batch_size=8,
            num_epochs=0,
            total_steps=50000,
            chkpt_steps=1000,
            growing_steps=50000,
            generator_learning_rate=9e-4, 
            epsilon=1.0e-7, 
            generator_beta1=0.9,
            generator_beta2=0.999,
            discriminator_learning_rate=9e-4, 
            discriminator_beta1=0.9,
            discriminator_beta2=0.999,
            mode_seeking_loss_weight=0.1,
            wgan_lambda=10.0,
            wgan_epsilon=0.001,
            wgan_target=1.0
        )

class Adam(object):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.trainable_variables, self.moment1, self.moment2, self.updates = {}, {}, {}, {}

    # a quick simple implementation of the adam algorithm, don't need any advanced features from the keras optimizers
    def update_variable(self, gradient: tf.Tensor, variable: tf.Variable):
        if variable.name not in self.trainable_variables:
            self.trainable_variables[variable.name] = variable
            self.moment1[variable.name] = tf.zeros_like(variable)
            self.moment2[variable.name] = tf.zeros_like(variable)
            self.updates[variable.name] = 0
        self.moment1[variable.name] = self.beta_1 * self.moment1[variable.name] + gradient * (1.0 - self.beta_1)
        self.moment2[variable.name] = self.beta_2 * self.moment2[variable.name] + tf.square(gradient) * (1.0 - self.beta_2)    
        corrected_moment1 = self.moment1[variable.name] / (1.0 - self.beta_1 ** (self.updates[variable.name] + 1))
        corrected_moment2 = self.moment2[variable.name] / (1.0 - self.beta_2 ** (self.updates[variable.name] + 1))
        variable.assign_sub(self.lr * corrected_moment1 / (tf.sqrt(corrected_moment2) + self.epsilon))
        self.updates[variable.name] += 1

    def apply_gradients(self, grads_and_vars: zip):
        for grad, var in grads_and_vars:
            if grad is not None:
                self.update_variable(grad, var)

class Network(object):
    def __init__(self, resolution=[128, 1024], fmap_base=1024, fmap_decay=1.0, fmap_max=256, **kwargs):
        self.growing_depth = tf.Variable(0.0, trainable=False)
        self.growing_level = 0
        self.min_resolution = np.array([2, 16])
        self.max_resolution = np.array(resolution)
        self.fmap_base = fmap_base
        self.fmap_max = fmap_max
        self.fmap_decay = fmap_decay # log2 feature map reduction
        self.min_depth = 0
        self.max_depth = int(max(np.log2(self.max_resolution // self.min_resolution)))
        self.optimizer = self.create_optimizer()

    def resolution(self, depth):
        # fancy version of self.min_resolution * 2 ** depth
        # network grows by powers of 2
        return self.min_resolution << depth

    def nf(self, depth):
        return min(int(self.fmap_base / (2.0 ** (depth * self.fmap_decay))), self.fmap_max)
    
    def create_optimizer(self) -> Adam:
        # gotta implement this in Generator and Discriminator
        pass

    def get_trainable_variables(self, graph_variables: dict) -> list:
        # also to implement
        pass

class Generator(Network):
    def __init__(self, resolution=[128, 1024], max_channels=256):
        super().__init__(resolution, fmap_max=max_channels)
        # note if i update batch size then i have to redefine the function to update x default arg
        self.fake_input_fn = lambda x=hyper_params['batch_size']: tf.random.normal(shape=[x, 256])
    
    def create_optimizer(self):
        return Adam(
            lr=hyper_params['generator_learning_rate'], 
            beta_1=hyper_params['generator_beta1'], 
            beta_2=hyper_params['generator_beta2'], 
            epsilon=hyper_params['epsilon'])
    
    def block(self, input, depth):
            if depth == self.min_depth:
                input = ops.pixel_normalization(input)
                input = ops.dense(input, self.nf(depth)*self.min_resolution.prod(), 'generator_dense')
                input = tf.reshape(tensor=input, shape=[-1, *self.min_resolution, self.nf(depth)])
                input = tf.nn.leaky_relu(input)
                input = ops.pixel_normalization(input)
                input = ops.conv2d(input, self.nf(depth), 'generator_conv2d_'+str(depth), (3, 3), padding='SAME')
                input = tf.nn.leaky_relu(input)
                input = ops.pixel_normalization(input)
            else:
                # karras uses upscaled convolution, this is another way to get a similar result
                input = ops.conv2d_transpose(input, self.nf(depth), 'generator_conv2d_transpose_'+str(depth), (3, 3), (2, 2), padding='SAME')
                input = tf.nn.leaky_relu(input)
                input = ops.pixel_normalization(input)
                input = ops.conv2d(input, self.nf(depth), 'generator_conv2d_'+str(depth), (3, 3), padding='SAME')
                input = tf.nn.leaky_relu(input)
                input = ops.pixel_normalization(input)
            return input
        
    def torgb(self, input, depth):
        return tf.nn.tanh(ops.conv2d(input, 2, 'generator_rgb_'+str(depth), (1, 1), padding='SAME', variance=1.0))
    
    def grow(self, input, depth):

        # lod = self.max_depth - depth # log2 resolution increase
        # y = self.block(input, depth)
        factor = self.max_resolution // self.resolution(depth)
        x = self.block(input, depth)
        img = lambda: ops.upscale2d(ops.lerp(self.torgb(x, depth), ops.upscale2d(self.torgb(input, depth-1)), depth-self.growing_depth), factor)
        if depth == self.min_depth:
            # keep growing the network until we are under the total network level
            return tf.cond(pred=self.growing_depth>depth, true_fn=lambda: self.grow(x, depth+1), false_fn=lambda: ops.upscale2d(self.torgb(x, depth), factor))
        elif depth == self.max_depth:
            # if we are at the max depth, then just colorize the image because it's already full size
            # otherwise get an image from this layer and interpolate it with an upscaled image from the last layer
            # this processes the transition from the penultimate to final layer
            return tf.cond(pred=self.growing_depth>depth, true_fn=lambda: self.torgb(x, depth), false_fn=img)
            # fn = lambda: ops.lerp(ops.upscale2d(self.torgb(input, depth-1)), ops.upscale2d(self.torgb(x, depth), factor), depth-self.growing_depth)
        else:
            # this processes all other transitions
            return tf.cond(pred=self.growing_depth>depth, true_fn=lambda: self.grow(x, depth+1), false_fn=img)
    
    def __call__(self, latents=None):
        if latents is None:
            # this is supposed to just be a random noise vector (a seed of sorts for generation)
            latents = self.fake_input_fn()
        return self.grow(latents, self.min_depth)
    
    def get_trainable_variables(self, graph_variables: dict) -> list:
        variables = []
        for name, var in graph_variables.items():
            if 'generator' in name and 'param' not in name:
                variables.append(var)
        return variables
    
class Discriminator(Network):
    def __init__(self, resolution=[128, 1024], max_channels=256):
        super().__init__(resolution, fmap_max=max_channels)
    
    def create_optimizer(self):
        return Adam(
            lr=hyper_params['discriminator_learning_rate'], 
            beta_1=hyper_params['discriminator_beta1'], 
            beta_2=hyper_params['discriminator_beta2'], 
            epsilon=hyper_params['epsilon'])

    def block(self, input, depth):
        if depth == self.min_depth:
            input = ops.conv2d(input, self.nf(depth), 'discriminator_conv2d_'+str(depth), (3, 3), padding='SAME')
            input = ops.flatten(input)
            input = ops.dense(input, self.nf(depth-1), 'discriminator_dense_0')
            input = tf.nn.leaky_relu(input)
            input = ops.dense(input, 1, 'discriminator_dense_1', variance=1.0)
            input = tf.nn.sigmoid(input)
        else:
            input = ops.conv2d(input, self.nf(depth), 'discriminator_conv2d_'+str(depth)+'a', (3, 3), padding='SAME')
            input = tf.nn.leaky_relu(input)
            input = ops.conv2d(input, self.nf(depth-1), 'discriminator_conv2d_'+str(depth)+'b', (3, 3), (2, 2), padding='SAME')
            input = tf.nn.leaky_relu(input)
        return input
        
    def fromrgb(self, input, depth):
        return tf.nn.leaky_relu(ops.conv2d(input, self.nf(depth), 'discriminator_rgb_'+str(depth), (1, 1), padding='SAME'))
    
    def grow(self, image, depth):
        factor = self.max_resolution // self.resolution(depth)
        x = self.fromrgb(ops.downscale2d(image, factor), depth)
        img = lambda: ops.lerp(self.block(x, depth), self.fromrgb(ops.downscale2d(image, factor*2), depth-1), depth-self.growing_depth)
        if depth == self.min_depth:
            # the outer block is just to get our final prediction, but keep growing the discriminator layers
            return tf.cond(pred=self.growing_depth>depth, true_fn=lambda: self.block(self.grow(image, depth+1), depth), false_fn=lambda: self.block(x, depth))
        elif depth == self.max_depth:
            # no next layer to transition to if the cond is true
            return tf.cond(pred=self.growing_depth>depth, true_fn=lambda: self.block(x, depth), false_fn=img)
        else:
            # keep growing the network
            return tf.cond(pred=self.growing_depth>depth, true_fn=lambda: self.block(self.grow(image, depth+1), depth), false_fn=img)
        
    def __call__(self, image):
        return self.grow(image, self.min_depth)
    
    def get_trainable_variables(self, graph_variables: dict) -> list:
        variables = []
        for name, var in graph_variables.items():
            if 'discriminator' in name and 'param' not in name:
                variables.append(var)
        return variables

class Model(object):
    def __init__(self, resolution=[128, 1024], max_channels=256, fetch_data=True):
        self.data_path = 'C:/Users/saraa/Desktop/music_generation/train_music/'
        self.out_path = 'C:/Users/saraa/Desktop/music_generation/generated_music/'
        self.chkpt_dir = 'C:/Users/saraa/Desktop/music_generation/checkpoints/'
        self.dataset = self.get_dataset() if fetch_data else None
        self.steps = 0
        self.g = Generator(resolution, max_channels)
        self.d = Discriminator(resolution, max_channels)
    def update_training_steps(self):
        self.steps += 1
        self.g.growing_level = self.steps / hyper_params['growing_steps']
        self.d.growing_level = self.steps / hyper_params['growing_steps']
        self.g.growing_depth = math.log2(1 + ((1 << (self.g.max_depth + 1)) - 1) * self.g.growing_level)
        self.d.growing_depth = math.log2(1 + ((1 << (self.d.max_depth + 1)) - 1) * self.d.growing_level)
    # cutoff is 41 (F2) to 89 (F6)
    def get_dataset(self) -> np.ndarray:
        files = sorted(os.listdir(self.data_path), key=lambda x: int(x[:-4]))
        num_samples = len(files)
        data = []
        for i in range(num_samples):
            waveform = tf.io.read_file(self.data_path+str(i)+'.wav')
            waveform, _ = tf.audio.decode_wav(contents=waveform, desired_channels=1, desired_samples=64000)
            data.append(waveform)
        # need to batch the data
        batched_data = []
        for i in range(0, len(data), hyper_params['batch_size']):
            if i+hyper_params['batch_size'] < len(data):
                batched_data.append(data[i:i+hyper_params['batch_size']])
        training_data = np.array(batched_data)
        print(f'\033[93mmemory used by training data: {training_data.nbytes / (1024 ** 3)} GB\033[00m')
        return training_data


    def learn(self, index):
        with tf.GradientTape(persistent=True) as tape:
            real_waveform = self.dataset[index] # dataset is already batched
            real_waveform = tf.squeeze(real_waveform, axis=-1)
            latents = self.g.fake_input_fn()
            fake_image = self.g(latents)
            real_spectrogram, real_instantaneous_frequency = spec_ops.convert_to_spectrogram(real_waveform, **spectral_params)
            real_image = tf.stack([real_spectrogram, real_instantaneous_frequency], axis=1) # along channel dimension
            real_image = tf.transpose(real_image, perm=[0, 2, 3, 1]) # convert from nchw to nhwc
            tape.watch(real_image)
            real_logits = self.d(real_image)
            fake_logits = self.d(fake_image)
            
            mean_fake = tf.reduce_mean(fake_logits)
            mean_real = tf.reduce_mean(real_logits)
            # using the regular GAN loss with log can make the gradient kinda unstable
            # thus we use wgan-gp as in karras but no ac-gan because no labels
            discriminator_loss = mean_fake - mean_real # maximize diff between output on real vs fake by minimizing the negative wasserstein loss
            # add gradient penalties to the discriminator loss as in wgan-gp loss
            mixing_factor = tf.random.uniform(shape=[hyper_params['batch_size'], 1, 1, 1], minval=0.0, maxval=1.0)
            mixed_image = mixing_factor * real_image + ((1.0 - mixing_factor) * fake_image) # essentially a lerp between the two
            tape.watch(mixed_image)
            mixed_logits = self.d(mixed_image)
            mixed_grad = tape.gradient(mixed_logits, mixed_image)
            mixed_norm = tf.sqrt(tf.reduce_sum(tf.square(mixed_grad), axis=[1,2,3])) # everything but the batch dim
            gradient_penalty = tf.square(mixed_norm - hyper_params['wgan_target'])
            discriminator_loss += gradient_penalty * (hyper_params['wgan_lambda'] / (hyper_params['wgan_target']**2))
            epsilon_penalty = tf.square(real_logits)
            discriminator_loss += epsilon_penalty * hyper_params['wgan_epsilon']
            # generator loss is wgan like in karras, but no ac-gan
            generator_loss = -mean_fake # maximize fake logits by minimizing the negative

        # now compute gradients with automatic differentiation
        generator_variables = self.g.get_trainable_variables(ops.vars)
        discriminator_variables = self.d.get_trainable_variables(ops.vars)
        generator_gradients = tape.gradient(generator_loss, generator_variables)
        discriminator_gradients = tape.gradient(discriminator_loss, discriminator_variables)
        # apply them with optimizers
        self.g.optimizer.apply_gradients(zip(generator_gradients, generator_variables))
        self.d.optimizer.apply_gradients(zip(discriminator_gradients, discriminator_variables))

    def generate_examples(self, n=1, filename='output'):
        latents = self.g.fake_input_fn(n)
        fake_image = tf.transpose(self.g(latents), perm=[0, 3, 1, 2]) # convert to nchw
        fake_spectrogram, fake_instantaneous_frequency = tf.unstack(fake_image, axis=1)
        fake_waveform = spec_ops.convert_to_waveform(fake_spectrogram, fake_instantaneous_frequency, **spectral_params)
        waveforms = tf.unstack(fake_waveform, axis=0)
        for waveform in waveforms:
            self.write_wav(waveform.numpy(), filename)

    # this is the training loop
    def train(self):
        if hyper_params['num_epochs'] == 0:
            hyper_params['num_epochs'] = hyper_params['total_steps'] // hyper_params['batch_size']
        for epoch in range(hyper_params['num_epochs']):
            print(f'begin epoch {epoch+1}')
            for example in range(self.dataset.shape[0]):
                self.learn(example)
                self.update_training_steps()
                if self.steps % 100 == 0:
                    print(f'{datetime.datetime.now()} completed {self.steps} training steps')
                if self.steps % hyper_params['chkpt_steps'] == 0:
                    ops.save(self.chkpt_dir)
                    self.generate_examples(4, f'gen{self.steps//hyper_params['chkpt_steps']}_')

    def write_wav(self, waveform, filename='output'):
        files = os.listdir(self.out_path)
        filename = filename+str(len(files))+'.wav'
        path = self.out_path+filename
        wavfile.write(path, 16000, waveform)


