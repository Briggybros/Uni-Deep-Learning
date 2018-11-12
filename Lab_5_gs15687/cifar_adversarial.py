from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cleverhans.attacks import FastGradientMethod
from cleverhans.model import CallableModelWrapper

import sys

import os
import os.path

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'labsheets', 'CIFAR10'))
import cifar10 as cf

FLAGS = tf.app.flags.FLAGS

# Runtime parameters
tf.app.flags.DEFINE_string('data-dir', os.getcwd() + '/dataset/',
                            'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('max-steps', 10000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log-frequency', 10,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model', 1000,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch-size', 256, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 1e-3, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate-decay', 0.8, 'Rate of decay of learning rate (default: %(default)d)')
tf.app.flags.DEFINE_float('fgsm-eps', 0.05, 'FSGM EPS value (default: %(default)d')


# Data parameters
tf.app.flags.DEFINE_integer('img-width', 32, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-height', 32, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-channels', 3, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num-classes', 10, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')



run_log_dir = os.path.join(FLAGS.log_dir,
                           'Lab_5_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size,
                                                        lr=FLAGS.learning_rate))

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='biases')

xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
def deepnn(x_image, train):
    """deepnn builds the graph for a deep net for classifying CIFAR10 images.

  Args:
      x: an input tensor with the dimensions (N_examples, 3072), where 3072 is the
        number of pixels in a standard CIFAR10 image.

  Returns:
      y: is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the object images into one of 10 classes
        (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
      img_summary: a string tensor containing sampled input images.
    """

    x_image = tf.cond(train, lambda: tf.map_fn(tf.image.random_flip_left_right, x_image), lambda: x_image)
    x_image = tf.cond(train, lambda: tf.map_fn(lambda x: tf.image.random_brightness(x, 0.5), x_image), lambda: x_image)

    img_summary = tf.summary.image('Input_images', x_image)

    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=32,
        kernel_size=[5,5],
        kernel_initializer=xavier_initializer,
        padding='same',
        use_bias=True,
        name='conv1'
    )
    conv1_bn = tf.nn.relu(tf.layers.batch_normalization(conv1, training=train, name="conv1_bn"))
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1_bn,
        pool_size=[2, 2],
        strides=2,
        name='pool1'
    )

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        kernel_initializer=xavier_initializer,
        padding='same',
        use_bias=True,
        name='conv2'
    )
    conv2_bn = tf.nn.relu(tf.layers.batch_normalization(conv2, training=train, name="conv2_bn"))
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2_bn,
        pool_size=[2, 2],
        strides=2,
        name='pool2'
    )

    v = tf.reshape(pool2, [-1, 4096])

    fc1 = tf.layers.dense(
        inputs=v,
        units=1024,
        activation=tf.nn.relu,
        use_bias=True,
        name='fc1'
    )

    fc2 = tf.layers.dense(
        inputs=fc1,
        units=1024,
        activation=tf.nn.relu,
        use_bias=True,
        name='fc2'
    )

    out = tf.layers.dense(
        inputs=fc2,
        units=10,
        activation=None,
        use_bias=True,
        name='out'
    )

    return out


def main(_):
    tf.reset_default_graph()

    # Import data
    cifar = cf.cifar10(batchSize=FLAGS.batch_size, downloadDir=FLAGS.data_dir)
    cifar.preprocess()

    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
        # Whether model is training
        train = tf.placeholder(tf.bool, [])

    # Reshape to use within a convolutional neural net.  Last dimension is for
    # 'features' - it would be 1 one for a grayscale image, 3 for an RGB image,
    # 4 for RGBA, etc.
    x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])

    with tf.variable_scope('model'):
        # Build the graph for the deep net
        y_conv = deepnn(x_image, train)
        model = CallableModelWrapper(lambda _x: deepnn(_x, train), 'logits')

    # Define your loss function - softmax_cross_entropy
    with tf.variable_scope('x_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    
    # Define your AdamOptimiser, using FLAGS.learning_rate to minimixe the loss function
    decayed_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, tf.Variable(0, trainable=False), 1000, FLAGS.learning_rate_decay)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimiser = tf.train.AdamOptimizer(decayed_learning_rate, name="Adam").minimize(cross_entropy)

    # calculate the prediction and the accuracy
    accuracy, acc_op = tf.metrics.accuracy(labels=tf.argmax(y_, axis=1), predictions=tf.argmax(y_conv, axis=1))
    adv_accuracy, adv_acc_op = tf.metrics.accuracy(labels=tf.argmax(y_, axis=1), predictions=tf.argmax(y_conv, axis=1))
    
    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)
    adv_acc_summary = tf.summary.scalar('Adv Accuracy', adv_accuracy)

    # summaries for TensorBoard visualisation
    training_summary = tf.summary.merge([loss_summary])
    test_summary = tf.summary.merge([acc_summary])
    adv_test_summary = tf.summary.merge([adv_acc_summary])

    # saver for checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.Session() as sess:
        with tf.variable_scope('model', reuse=True):
            fgsm = FastGradientMethod(model, sess=sess)
            adv_test = fgsm.generate(x_image, eps=FLAGS.fgsm_eps, clip_min=0.0, clip_max=1.0)

            test_img_summary = tf.summary.image('Test Images', x_image)
            adv_test_img_summary = tf.summary.image('Adversarial test Images', adv_test)
            adv_summary = tf.summary.merge([test_img_summary, adv_test_img_summary])

        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph, flush_secs=5)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir + '_validate', sess.graph, flush_secs=5)
        adversarial_writer = tf.summary.FileWriter(run_log_dir + "_adversarial", sess.graph, flush_secs=5)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Training and validation
        for step in range(FLAGS.max_steps):
            # Training: Backpropagation using train set
            (trainImages, trainLabels) = cifar.getTrainBatch()
            (testImages, testLabels) = cifar.getTestBatch()
            
            _, summary_str = sess.run([optimiser, training_summary], feed_dict={x: trainImages, y_: trainLabels, train: True})

            
            if step % (FLAGS.log_frequency + 1) == 0:
                summary_writer.add_summary(summary_str, step)

            ## Validation: Monitoring accuracy using validation set
            if step % FLAGS.log_frequency == 0:
                accuracy, summary_str, adv_images = sess.run([acc_op, test_summary, adv_test], feed_dict={x: testImages, y_: testLabels, train: False})
                adv_accuracy, adv_summary_str = sess.run([adv_acc_op, adv_test_summary], feed_dict={x_image: adv_images, y_: testLabels, train: False})
                print('step %d, accuracy on validation batch: %g' % (step, accuracy))
                summary_writer_validation.add_summary(summary_str, step)
                adversarial_writer.add_summary(adv_summary_str, step)

            ## Save the model checkpoint periodically.
            if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(run_log_dir + '_train', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        # Testing

        # resetting the internal batch indexes
        cifar.reset()
        evaluated_images = 0
        test_accuracy = 0
        adv_test_accuracy = 0
        batch_count = 0

        # don't loop back when we reach the end of the test set
        while evaluated_images != cifar.nTestSamples:
            (testImages, testLabels) = cifar.getTestBatch(allowSmallerBatches=True)
            test_accuracy_temp, _, adv_images = sess.run([acc_op, test_summary, adv_test], feed_dict={x: testImages, y_: testLabels, train: False})
            adv_test_accuracy_temp = sess.run(adv_acc_op, feed_dict={x_image: adv_images, y_: testLabels, train: False})

            batch_count = batch_count + 1
            test_accuracy = test_accuracy + test_accuracy_temp
            adv_test_accuracy = adv_test_accuracy + adv_test_accuracy_temp
            evaluated_images = evaluated_images + testLabels.shape[0]

        test_accuracy = test_accuracy / batch_count
        adv_test_accuracy = adv_test_accuracy / batch_count
        print('test set: accuracy on test set: %0.3f' % test_accuracy)
        print('test set: accuracy on adversarial test set: %0.3f' % adv_test_accuracy)



if __name__ == '__main__':
    tf.app.run(main=main)
