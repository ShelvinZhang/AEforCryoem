import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as io

from Model import VariationalAutoencoder
import Utils
import mrcfile

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '', 'Directory for storing data')
flags.DEFINE_string('file_name', 'nonoisectf_stack1000.mrcs', 'Data file name')
flags.DEFINE_integer('training_epochs', 10, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size')
flags.DEFINE_float('starter_learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('decay_epochs', 2000, 'Learning rate decay epochs')
flags.DEFINE_float('decay_rate', 0.1, 'Learning rate decay rate')
flags.DEFINE_integer('display_step', 1, 'Display step')
flags.DEFINE_list('plot_step', [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000], 'Step to plot the picture')
flags.DEFINE_integer('n_plot', 5, 'Number of pictures to plot')
flags.DEFINE_string('summaries_dir', './summary', 'Summaries directory')

flags.DEFINE_integer('image_size', 112, '')
flags.DEFINE_integer('n_samples', 1000, '')
flags.DEFINE_integer('n_hidden', 10, '')
flags.DEFINE_list('conv_strides', [2, 2, 2, 4], '')
flags.DEFINE_list('conv_features', [16 * x for x in [1, 2, 4, 4]], '')
flags.DEFINE_list('deconv_strides', [4, 2, 2, 2], '')
flags.DEFINE_list('deconv_features', [16 * x for x in [4, 4, 2, 1]], '')
flags.DEFINE_integer('n_encoder1', 200, '')


def train_vae(*args):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    # data load and preprocessing
    x_train = mrcfile.open(FLAGS.data_dir + FLAGS.file_name).data
    np.random.seed()
    order = np.array(range(FLAGS.n_samples))
    np.random.shuffle(order)
    x_train = Utils.zscore(x_train.reshape([FLAGS.n_samples, -1]), axis=0, truncate=True, scale=True)[order].reshape([-1, FLAGS.image_size, FLAGS.image_size])
    total_batch = int(FLAGS.n_samples / FLAGS.batch_size)
    vae = VariationalAutoencoder(FLAGS)

    # start training
    fig_reconstruction, axs_reconstruction = plt.subplots(2, FLAGS.n_plot)
    for epoch in range(FLAGS.training_epochs):
        avg_cost = 0.
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = Utils.get_random_block_from_data(x_train.data, FLAGS.batch_size)

            # Fit training using batch data
            cost, lr = vae.partial_fit(batch_xs, epoch, i, total_batch)
            # Compute average loss
            avg_cost += cost / FLAGS.n_samples * FLAGS.batch_size

        # Display logs per epoch step
        if epoch % FLAGS.display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost), " lr=%f" % lr)
            # Plot example reconstructions
            if epoch+1 in FLAGS.plot_step:
                plot_x = x_train.data[:FLAGS.n_plot]
                plot_recon = vae.reconstruct(plot_x)
                for example_i in range(FLAGS.n_plot):
                    axs_reconstruction[0][example_i].imshow(
                        np.reshape(plot_x[example_i, :], (FLAGS.image_size, FLAGS.image_size)),
                        cmap='gray')
                    axs_reconstruction[1][example_i].imshow(
                        np.reshape(plot_recon[example_i, ...], (FLAGS.image_size, FLAGS.image_size)),
                        cmap='gray')
                    axs_reconstruction[0][example_i].axis('off')
                    axs_reconstruction[1][example_i].axis('off')
                fig_reconstruction.savefig('reconstruction_%d.png' % (epoch+1))
                z_mean, z_log_sigma, z = vae.encode(x_train.data)
                io.savemat('z_%d' % (epoch+1), {'z_mean': z_mean, 'z_log_sigma': z_log_sigma, 'z': z})

    io.savemat('order', {'order': order})
    print("Total cost: " + str(vae.calc_total_cost(x_train.data)))


if __name__ == '__main__':
    tf.app.run(main=train_vae)