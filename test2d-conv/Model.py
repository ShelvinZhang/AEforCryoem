import tensorflow as tf
import Utils


class VariationalAutoencoder(object):

    def __init__(self, model_args):
        args = model_args
        decay_steps = args.decay_epochs * int(args.n_images / args.batch_size)
        self.encoder = {}
        self.decoder = {}

        # encode
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, args.image_size, args.image_size, 1], name='x-input')

        with tf.name_scope('encoder'):
            for conv_index in range(len(args.conv_strides)):
                layer = 'Conv' + str(conv_index+1)
                if not conv_index:
                    self.encoder[layer] = self._conv_block(self.x, args.conv_size[conv_index], 1, args.conv_features[conv_index],
                                                           args.conv_strides[conv_index], layer_num=2, act=tf.nn.selu, block_name=layer)
                else:
                    formerlayer = 'Conv' + str(conv_index)
                    self.encoder[layer] = self._conv_block(self.encoder[formerlayer], args.conv_size[conv_index],
                                                           args.conv_features[conv_index-1], args.conv_features[conv_index],
                                                           args.conv_strides[conv_index], layer_num=2, act=tf.nn.selu, block_name=layer)
            # flatten
            encoder_shape = self.encoder[layer].get_shape().as_list()
            encoder_flat_shape = encoder_shape[1] * encoder_shape[2] * encoder_shape[3]
            self.encoder['flat'] = tf.reshape(self.encoder[layer], [-1, encoder_flat_shape])

        with tf.name_scope('code'):
            self.z_mean = self._nn_layer(self.encoder['flat'], encoder_flat_shape, args.n_code, act=tf.nn.tanh, layer_name='z_mean')
            tf.summary.histogram('z_mean', self.z_mean)

            self.z_log_sigma = self._nn_layer(self.encoder['flat'], encoder_flat_shape, args.n_code, act=tf.nn.tanh, layer_name='z_log_sigma')
            tf.summary.histogram('z_mean', self.z_mean)

            # sample from gaussian distribution
            with tf.name_scope('epsilon'):
                eps = tf.random_normal(tf.stack([tf.shape(self.z_log_sigma)[0], args.n_code]), 0, 1, dtype=tf.float32)
            with tf.name_scope('z'):
                self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma)), eps))

        # decode
        with tf.name_scope('decoder'):
            self.decoder['fold'] = tf.reshape(self._nn_layer(self.z, args.n_code, encoder_shape[1]*encoder_shape[2], act=tf.nn.tanh, layer_name='deConv0'),
                                              [-1, encoder_shape[1], encoder_shape[2], 1])
            for deconv_index in range(len(args.deconv_strides)):
                layer = 'deConv' + str(deconv_index+1)
                if not deconv_index:
                    self.decoder[layer] = self._deconv_block(self.decoder['fold'], args.deconv_size[deconv_index], 1, args.deconv_features[deconv_index],
                                                             args.deconv_strides[deconv_index], layer_num=2, act=tf.nn.selu, block_name=layer)
                else:
                    formerlayer = 'deConv' + str(deconv_index)
                    self.decoder[layer] = self._deconv_block(self.decoder[formerlayer], args.deconv_size[deconv_index],
                                                             args.deconv_features[deconv_index-1], args.deconv_features[deconv_index],
                                                             args.deconv_strides[deconv_index], layer_num=2, act=tf.nn.selu, block_name=layer)

        with tf.name_scope('x-output'):
            self.reconstruction = self._deconv_layer(self.decoder[layer], args.deconv_size[deconv_index], args.deconv_features[deconv_index], 1,
                                                     strides=[1, 1, 1, 1], act=tf.nn.tanh, layer_name='x-output')

        # cost
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                # reconstruction_loss = -tf.reduce_sum(self.x * tf.log(self.reconstruction + 1e-8) +
                #                                     (1 - self.x) * tf.log(1 - self.reconstruction + 1e-8), 1)
                self.reconstruction_loss = tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
                tf.summary.scalar('loss_recon', self.reconstruction_loss)
            with tf.name_scope('kl_divergence'):
                self.latent_loss = -0.5 * tf.reduce_mean(1.0 + 2.0 * self.z_log_sigma
                                                         - tf.square(self.z_mean)
                                                         - tf.exp(2.0 * self.z_log_sigma), 1)
                # tf.summary.scalar('loss_kl', self.latent_loss)
            with tf.name_scope('total'):
                self.cost = tf.reduce_mean(self.reconstruction_loss + self.latent_loss)
                tf.summary.scalar('loss_function', self.cost)

        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(args.starter_learning_rate, global_step, decay_steps, args.decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost, global_step=global_step)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(args.summaries_dir, self.sess.graph)
        self.sess.run(init)

    def _weight_variable(self, shape):
        '''Helper function to create a weight variable initialized with
        a normal distribution

        Parameters
        ----------
        shape : list
            Size of weight variable
        '''
        # Xavier initialization
        initial = Utils.xavier_initializer()(shape)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        '''Helper function to create a bias variable initialized with
        a constant value.

        Parameters
        ----------
        shape : list
            Size of weight variable
        '''
        initial = tf.truncated_normal(shape, mean=0.0, stddev=0.01)
        return tf.Variable(initial)

    def _variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar('sttdev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)

    def _nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.elu):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self._weight_variable([input_dim, output_dim])
                # _variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                biases = self._bias_variable([output_dim])
                # _variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                # tf.summary.histogram(layer_name + '/pre_activations', preactivate)
            if not act:
                return preactivate
            else:
                activations = act(preactivate, 'activation')
                # tf.summary.histogram(layer_name + '/activations', activations)
                return activations

    def _conv_layer(self, input_tensor, filter_size, in_channels, out_channels, strides=[1, 1, 1, 1], padding='SAME', act=tf.nn.elu, layer_name=''):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self._weight_variable([filter_size, filter_size, in_channels, out_channels])
                # _variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                biases = self._bias_variable([out_channels])
                # _variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.nn.conv2d(input_tensor, weights, strides, padding) + biases
                # tf.summary.histogram(layer_name + '/pre_activations', preactivate)
            activations = act(preactivate, 'activation')
            # tf.summary.histogram(layer_name + '/activations', activations)
            return activations

    def _conv_block(self, input_tensor, filter_size, in_channels, out_channels, pool_stride, layer_num=3, padding='SAME', act=tf.nn.elu, block_name=''):
        with tf.name_scope(block_name):
            for index in range(layer_num):
                if not index:
                    output = self._conv_layer(input_tensor, filter_size, in_channels, out_channels, strides=[1, 1, 1, 1],
                                              padding=padding, act=act, layer_name=block_name+'_'+str(index+1))
                else:
                    output = self._conv_layer(output, filter_size, out_channels, out_channels, strides=[1, pool_stride, pool_stride, 1],
                                              padding=padding, act=act, layer_name=block_name+'_'+str(index+1))
            return output

    def _deconv_layer(self, input_tensor, filter_size, in_channels, out_channels, strides=[1, 2, 2, 1], padding='SAME', act=tf.nn.elu, layer_name=''):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self._weight_variable([filter_size, filter_size, in_channels, out_channels])
                # _variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                biases = self._bias_variable([out_channels])
                # _variable_summaries(biases, layer_name + '/biases')
            input_size = input_tensor.shape.as_list()
            output_shape = [tf.shape(input_tensor)[0], input_size[1]*strides[1], input_size[2]*strides[2], out_channels]
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.nn.conv2d_transpose(input_tensor, tf.transpose(weights, perm=[0, 1, 3, 2]), output_shape, strides, padding) + biases
                # tf.summary.histogram(layer_name + '/pre_activations', preactivate)
            activations = act(preactivate, 'activation')
            # tf.summary.histogram(layer_name + '/activations', activations)
            return activations

    def _deconv_block(self, input_tensor, filter_size, in_channels, out_channels, pool_stride, layer_num=3, padding='SAME', act=tf.nn.elu, block_name=''):
        with tf.name_scope(block_name):
            for index in range(layer_num):
                if not index:
                    output = self._deconv_layer(input_tensor, filter_size, in_channels, out_channels, strides=[1, 1, 1, 1],
                                                padding=padding, act=act, layer_name=block_name+'_'+str(index+1))
                else:
                    output = self._deconv_layer(output, filter_size, out_channels, out_channels, strides=[1, pool_stride, pool_stride, 1],
                                                padding=padding, act=act, layer_name=block_name+'_'+str(index+1))
            return output

    def _pool_layer(self, input_tensor, filter_size, strides=[1, 2, 2, 1], padding='SAME', layer_name=''):
        with tf.name_scope(layer_name):
            return tf.nn.max_pool(input_tensor, filter_size, strides, padding)

    def partial_fit(self, x, epoch, step, total_batch):
        summary, cost, opt, lr = self.sess.run((self.merged, self.cost, self.optimizer, self.learning_rate),
                                               feed_dict={self.x: x})
        if step == total_batch-1:
            self.train_writer.add_summary(summary, epoch)
        return cost, lr

    def calc_total_cost(self, x):
        saver = tf.train.Saver()
        cost = self.sess.run(self.cost, feed_dict={self.x: x})
        saver.save(self.sess, 'model.ckpt')
        return cost

    def reconstruct(self, x):
        return self.sess.run(self.reconstruction, feed_dict={self.x: x})

    def generate(self, code=None):
        # if code is None:
        #     code = np.random.normal(size=self.weights["b1"])
        # return self.sess.run(self.reconstruction, feed_dict={self.z_mean: code})
        return self.sess.run(self.reconstruction, feed_dict={self.z: code})

    def encode(self, x):
        return self.sess.run([self.z_mean, self.z_log_sigma, self.z], feed_dict={self.x: x})

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, "./model.ckpt")
        print("Model restored.")

    '''
    def transform(self, X):
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])
    '''
