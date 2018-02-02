import tensorflow as tf
from .ops import linear, conv2d, conv2d_transpose, lrelu

class dcgan(object):
    def __init__(self, output_size=64, batch_size=64, 
                 gradient_penalty_mode=True, gradient_penalty_lambda=10.,
                 nd_layers=4, ng_layers=4, df_dim=128, gf_dim=128, 
                 c_dim=1, z_dim=100, data_format="NHWC",
                 gen_prior=tf.random_normal, transpose_b=False):

        self.output_size = output_size
        self.batch_size = batch_size
        self.gradient_penalty_mode = gradient_penalty_mode
        self.gradient_penalty_lambda = gradient_penalty_lambda
        self.nd_layers = nd_layers
        self.ng_layers = ng_layers
        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.data_format = data_format
        self.gen_prior = gen_prior
        self.transpose_b = transpose_b # transpose weight matrix in linear layers for (possible) better performance when running on HSW/KNL
        self.stride = 2 # this is fixed for this architecture

        self._check_architecture_consistency()


        self.batchnorm_kwargs = {'epsilon' : 1e-5, 'decay': 0.9, 
                                 'updates_collections': None, 'scale': True,
                                 'fused': True, 'data_format': self.data_format}

    def training_graph(self):

        with tf.variable_scope("counters") as counters_scope:
            self.epoch = tf.Variable(-1, name='epoch', trainable=False)
            self.increment_epoch = tf.assign(self.epoch, self.epoch+1)
            self.global_step = tf.train.get_or_create_global_step()

        if self.data_format == "NHWC":
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='real_images')
        else:
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.c_dim, self.output_size, self.output_size], name='real_images')

        self.z = self.gen_prior(shape=[self.batch_size, self.z_dim])

        with tf.variable_scope("critic") as c_scope:
            mean_critic_scores_real = tf.reduce_mean(self.critic(self.images, is_training=True))

        with tf.variable_scope("generator") as g_scope:
            g_images = self.generator(self.z, is_training=True)

        with tf.variable_scope("critic") as c_scope:
            c_scope.reuse_variables()
            mean_critic_scores_fake = tf.reduce_mean(self.critic(g_images, is_training=True))

            if self.gradient_penalty_mode:
                epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], minval=0., maxval=1.)
                interpolated = g_images + epsilon * (images - g_images)
                critic_scores_interpolated = self.critic(interpolated, is_training=True)

        with tf.name_scope("losses"):
            with tf.name_scope("critic"):

                self.c_loss = -mean_critic_scores_real + mean_critic_scores_fake

                if self.gradient_penalty_mode:
                    gradients = tf.gradients(critic_scores_interpolated, [interpolated])[0]
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
                    gradient_penalty = tf.reduce_mean(tf.square(slopes-1.))

                    gradient_penalty_rate = tf.train.exponential_decay(self.gradient_penalty_lambda, self.global_step, 40000, 0.96)
                    self.c_loss += gradient_penalty_rate * gradient_penalty


            with tf.name_scope("generator"):
                self.g_loss = -mean_critic_scores_fake

        self.c_summary = tf.summary.merge([tf.summary.scalar("loss/mean_critic_scores_real", mean_critic_scores_real),
                                           tf.summary.scalar("loss/mean_critic_scores_fake", mean_critic_scores_fake),
                                           tf.summary.scalar("loss/critic_loss", self.c_loss)])

        if self.gradient_penalty_mode:
            self.c_summary = tf.summary.merge([self.d_summary,
                                               tf.summary.scalar("loss/gradient_penalty_term", gradient_penalty_rate * gradient_penalty)])

        g_sum = [tf.summary.scalar("loss/g", self.g_loss)]
        if self.data_format == "NHWC": # tf.summary.image is not implemented for NCHW
            g_sum.append(tf.summary.image("G", g_images, max_outputs=4))
        self.g_summary = tf.summary.merge(g_sum)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'critic/' in var.name]
        self.g_vars = [var for var in t_vars if 'generator/' in var.name]

        self.saver = tf.train.Saver(max_to_keep=8000)

    def inference_graph(self):

        if self.data_format == "NHWC":
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='real_images')
        else:
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.c_dim, self.output_size, self.output_size], name='real_images')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        with tf.variable_scope("discriminator") as d_scope:
            self.D = self.critic(self.images, is_training=False)

        with tf.variable_scope("generator") as g_scope:
            self.G = self.generator(self.z, is_training=False)

        with tf.variable_scope("counters") as counters_scope:
            self.epoch = tf.Variable(-1, name='epoch', trainable=False)
            self.increment_epoch = tf.assign(self.epoch, self.epoch+1)
            self.global_step = tf.train.get_or_create_global_step()

        self.saver = tf.train.Saver(max_to_keep=8000)

    def optimizer(self, learning_rate, beta1):

        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                                         .minimize(self.d_loss, var_list=self.d_vars, global_step=self.global_step)

        g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                                         .minimize(self.g_loss, var_list=self.g_vars)

        return tf.group(d_optim, g_optim, name="all_optims")

                                                   
    def generator(self, z, is_training):

        map_size = self.output_size/int(2**self.ng_layers)
        num_channels = self.gf_dim * int(2**(self.ng_layers -1))

        # h0 = relu(BN(reshape(FC(z))))
        z_ = linear(z, num_channels*map_size*map_size, 'h0_lin', transpose_b=self.transpose_b)
        h0 = tf.reshape(z_, self._tensor_data_format(-1, map_size, map_size, num_channels))
        bn0 = tf.contrib.layers.batch_norm(h0, is_training=is_training, scope='bn0', **self.batchnorm_kwargs)
        h0 = tf.nn.relu(bn0)

        chain = h0
        for h in range(1, self.ng_layers):
            # h1 = relu(BN(conv2d_transpose(h0)))
            map_size *= self.stride
            num_channels /= 2
            chain = conv2d_transpose(chain,
                                     self._tensor_data_format(self.batch_size, map_size, map_size, num_channels),
                                     stride=self.stride, data_format=self.data_format, name='h%i_conv2d_T'%h)
            chain = tf.contrib.layers.batch_norm(chain, is_training=is_training, scope='bn%i'%h, **self.batchnorm_kwargs)
            chain = tf.nn.relu(chain)

        # h1 = conv2d_transpose(h0)
        map_size *= self.stride
        hn = conv2d_transpose(chain,
                              self._tensor_data_format(self.batch_size, map_size, map_size, self.c_dim),
                              stride=self.stride, data_format=self.data_format, name='h%i_conv2d_T'%(self.ng_layers))

        return tf.nn.tanh(hn)


    def critic(self, image, is_training):

        # h0 = lrelu(conv2d(image))
        h0 = lrelu(conv2d(image, self.df_dim, self.data_format, name='h0_conv'))

        chain = h0
        for h in range(1, self.nd_layers):
            # h1 = lrelu(BN(conv2d(h0)))
            chain = conv2d(chain, self.df_dim*(2*h), self.data_format, name='h%i_conv'%h)
            if not self.gradient_penalty_mode: 
                chain = tf.contrib.layers.batch_norm(chain, is_training=is_training, scope='bn%i'%h, **self.batchnorm_kwargs)
            chain = lrelu(chain)

        # h1 = linear(reshape(h0))
        hn = linear(tf.reshape(chain, [self.batch_size, -1]), 1, 'h%i_lin'%self.nd_layers, transpose_b=self.transpose_b)

        return hn

    def _tensor_data_format(self, N, H, W, C):
        if self.data_format == "NHWC":
            return [int(N), int(H), int(W), int(C)]
        else:
            return [int(N), int(C), int(H), int(W)]

    def _check_architecture_consistency(self):

        if self.output_size/2**self.nd_layers < 1:
            print("Error: Number of discriminator conv. layers are larger than the output_size for this architecture")
            exit(0)

        if self.output_size/2**self.ng_layers < 1:
            print("Error: Number of generator conv_transpose layers are larger than the output_size for this architecture")
            exit(0)
