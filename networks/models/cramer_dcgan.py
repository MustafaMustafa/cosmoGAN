import tensorflow as tf
from .ops import linear, conv2d, conv2d_transpose, lrelu

class cramer_dcgan(object):
    def __init__(self, output_size=64, batch_size=64, 
                 nd_layers=4, ng_layers=4, df_dim=128, gf_dim=128, 
                 c_dim=1, z_dim=100, d_out_dim=256, gradient_lambda=10., 
                 data_format="NHWC",
                 gen_prior=tf.random_normal, transpose_b=False):

        self.output_size = output_size
        self.batch_size = batch_size
        self.nd_layers = nd_layers
        self.ng_layers = ng_layers
        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.d_out_dim = d_out_dim
        self.gradient_lambda = gradient_lambda
        self.data_format = data_format
        self.gen_prior = gen_prior
        self.transpose_b = transpose_b # transpose weight matrix in linear layers for (possible) better performance when running on HSW/KNL
        self.stride = 2 # this is fixed for this architecture

        self._check_architecture_consistency()

    def critic(self, x, xgp):
        h = self.discriminator
        return tf.norm(h(x) - h(xgp), axis=1) - tf.norm(h(x), axis=1)

    def training_graph(self):

        if self.data_format == "NHWC":
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='real_images')
        else:
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.c_dim, self.output_size, self.output_size], name='real_images')

        x = self.images
        self.z = self.gen_prior(shape=[self.batch_size, self.z_dim], dtype=tf.float32)
        self.zp = self.gen_prior(shape=[self.batch_size, self.z_dim], dtype=tf.float32)
        xg = self.generator(self.z)
        xgp = self.generator(self.zp)

        with tf.name_scope("losses"):
            with tf.name_scope("L_surrogate"):
                self.L_surrogate = tf.reduce_mean(self.critic(x, xgp) - self.critic(xg, xgp))
            with tf.name_scope("L_critic"):
                epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], minval=0., maxval=1., dtype=tf.float32)
                x_hat = epsilon * x + (1-epsilon) * xg
                f_x_hat = self.critic(x_hat, xgp)
                f_x_hat_gradient = tf.gradients(f_x_hat, x_hat)[0]
                gradient_penalty = tf.reduce_mean(self.gradient_lambda * tf.square(tf.norm(f_x_hat_gradient, axis=1) - 1))
                self.L_critic = -self.L_surrogate + gradient_penalty

        self.d_summary = tf.summary.merge([tf.summary.histogram("loss/L_critic", self.L_critic),
                                           tf.summary.histogram("loss/gradient_penalty", gradient_penalty)])

        g_sum = [tf.summary.scalar("loss/L_surrogate", self.L_surrogate)]

        if self.data_format == "NHWC": # tf.summary.image is not implemented for NCHW
            g_sum.append(tf.summary.image("G", xg, max_outputs=4))
        self.g_summary = tf.summary.merge(g_sum)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator/' in var.name]
        self.g_vars = [var for var in t_vars if 'generator/' in var.name]

        with tf.variable_scope("counters") as counters_scope:
            self.epoch = tf.Variable(-1, name='epoch', trainable=False)
            self.increment_epoch = tf.assign(self.epoch, self.epoch+1)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.saver = tf.train.Saver(max_to_keep=8000)

    def inference_graph(self):

        if self.data_format == "NHWC":
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='real_images')
        else:
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.c_dim, self.output_size, self.output_size], name='real_images')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.discriminator(self.images)
        self.G = self.generator(self.z)

        with tf.variable_scope("counters") as counters_scope:
            self.epoch = tf.Variable(-1, name='epoch', trainable=False)
            self.increment_epoch = tf.assign(self.epoch, self.epoch+1)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.saver = tf.train.Saver(max_to_keep=8000)

    def optimizer(self, learning_rate, beta1):

        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                                         .minimize(self.L_critic, var_list=self.d_vars, global_step=self.global_step)

        g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                                         .minimize(self.L_surrogate, var_list=self.g_vars)

        return tf.group(d_optim, g_optim, name="all_optims")

                                                   
    def generator(self, z):

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as g_scope:

            map_size = self.output_size/int(2**self.ng_layers)
            num_filters = self.gf_dim * int(2**(self.ng_layers -1))

            # h0 = relu(reshape(FC(z)))
            z_ = linear(z, num_filters*map_size*map_size, 'h0_lin', transpose_b=self.transpose_b)
            h0 = tf.reshape(z_, self._tensor_data_format(-1, map_size, map_size, num_filters))
            h0 = tf.nn.relu(h0)

            chain = h0
            for h in range(1, self.ng_layers):
                # h1 = relu(conv2d_transpose(h0))
                map_size *= self.stride
                num_filters /= 2
                chain = conv2d_transpose(chain,
                                         self._tensor_data_format(self.batch_size, map_size, map_size, num_filters),
                                         stride=self.stride, data_format=self.data_format, name='h%i_conv2d_T'%h)
                chain = tf.nn.relu(chain)

            # h1 = conv2d_transpose(h0)
            map_size *= self.stride
            hn = conv2d_transpose(chain,
                                  self._tensor_data_format(self.batch_size, map_size, map_size, self.c_dim),
                                  stride=self.stride, data_format=self.data_format, name='h%i_conv2d_T'%(self.ng_layers))

            return tf.nn.tanh(hn)


    def discriminator(self, image):

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as d_scope:

            chain = image
            for h in range(1, self.nd_layers):
                # h1 = lrelu(conv2d(h0))
                num_filters = self.df_dim if h==0 else self.df_dim*h*2
                chain = lrelu(conv2d(chain, num_filters, self.data_format, name='h%i_conv'%h))

            # h1 = linear(reshape(h0))
            hn = linear(tf.reshape(chain, [self.batch_size, -1]), self.d_out_dim, 'h%i_lin'%self.nd_layers, transpose_b=self.transpose_b)

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
