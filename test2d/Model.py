import tensorflow as tf
import numpy as np
import Utils


class VariationalAutoencoder(object):

    def __init__(self, model_args):
        self.args = model_args
        self.decay_steps = self.args.decay_epochs * int(self.args.n_samples / self.args.batch_size)

        # model
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, self.args.n_input**2], name='x-input')

        with tf.name_scope('encoder'):
            self.h_enc1 = self.nn_layer(self.x, self.args.n_input**2, self.args.n_encoder1, 'encoder1', act=eval(self.args.activation))
            self.h_enc2 = self.nn_layer(self.h_enc1, self.args.n_encoder1, self.args.n_encoder2, 'encoder1', act=eval(self.args.activation))
            self.h_enc3 = self.nn_layer(self.h_enc2, self.args.n_encoder2, self.args.n_encoder3, 'encoder1', act=eval(self.args.activation))

        with tf.name_scope('z_mean'):
            with tf.name_scope('weights'):
                w_mu = self.weight_variable([self.args.n_encoder3, self.args.n_hidden])
                # variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                b_mu = self.bias_variable([self.args.n_hidden])
                # variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                self.z_mean = tf.matmul(self.h_enc3, w_mu) + b_mu
                tf.summary.histogram('z_mean', self.z_mean)

        with tf.name_scope('z_log_sigma'):
            with tf.name_scope('weights'):
                w_log_sigma = self.weight_variable([self.args.n_encoder3, self.args.n_hidden])
                # variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                b_log_sigma = self.bias_variable([self.args.n_hidden])
                # variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                self.z_log_sigma = tf.matmul(self.h_enc3, w_log_sigma) + b_log_sigma
                tf.summary.histogram('z_log_sigma', self.z_log_sigma)

        # sample from gaussian distribution
        with tf.name_scope('epsilon'):
            eps = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.args.n_hidden]), 0, 1, dtype=tf.float32)
        with tf.name_scope('z'):
            self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma)), eps))

        with tf.name_scope('decoder'):
            self.h_dec1 = self.nn_layer(self.z, self.args.n_hidden, self.args.n_decoder1, 'decoder1', act=eval(self.args.activation))
            self.h_dec2 = self.nn_layer(self.h_dec1, self.args.n_decoder1, self.args.n_decoder2, 'decoder2', act=eval(self.args.activation))
            self.h_dec3 = self.nn_layer(self.h_dec2, self.args.n_decoder2, self.args.n_decoder3, 'decoder3', act=eval(self.args.activation))

        with tf.name_scope('x-output'):
            self.reconstruction = self.nn_layer(self.h_dec3, self.args.n_decoder3, self.args.n_input**2, 'x-output', act=tf.nn.sigmoid)

        # cost
        # reconstruction_loss = 0.5 * tf.reduce_sum(tf.pow(tf.sub(self.reconstruction, self.x), 2.0))
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                reconstruction_loss = -tf.reduce_sum(self.x * tf.log(self.reconstruction + 1e-8) +
                                                     (1 - self.x) * tf.log(1 - self.reconstruction + 1e-8), 1)
            with tf.name_scope('kl_divergence'):
                latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * self.z_log_sigma
                                                   - tf.square(self.z_mean)
                                                   - tf.exp(2.0 * self.z_log_sigma), 1)
            with tf.name_scope('total'):
                self.cost = tf.reduce_mean(reconstruction_loss + latent_loss)
                tf.summary.scalar('loss_function', self.cost)

        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            self.args.starter_learning_rate, global_step, self.decay_steps, self.args.decay_rate, staircase=True)
        self.optimizer = eval(self.args.optimizer)(self.learning_rate).minimize(self.cost, global_step=global_step)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.args.summaries_dir, self.sess.graph)
        self.sess.run(init)

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

    def generate(self, hidden=None):
        # if hidden is None:
        #     hidden = np.random.normal(size=self.weights["b1"])
        # return self.sess.run(self.reconstruction, feed_dict={self.z_mean: hidden})
        return self.sess.run(self.reconstruction, feed_dict={self.z: hidden})

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

    def weight_variable(self, shape):
        '''Helper function to create a weight variable initialized with
        a normal distribution

        Parameters
        ----------
        shape : list
            Size of weight variable
        '''
        # Gaussian initialization
        # initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
        # Xavier initialization
        initial = Utils.xavier_init(shape, constant=1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        '''Helper function to create a bias variable initialized with
        a constant value.

        Parameters
        ----------
        shape : list
            Size of weight variable
        '''
        initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
        return tf.Variable(initial)

    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)

    def nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.elu):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])
                # variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
                # variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                # tf.histogram_summary(layer_name + '/pre_activations', preactivate)
            activations = act(preactivate, 'activation')
            # tf.histogram_summary(layer_name + '/activations', activations)
            return activations
