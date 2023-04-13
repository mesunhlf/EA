# coding=utf-8
"""Implementation of SI-NI-FGSM attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.stats as st
from scipy.misc import imread, imsave

from utils import *
import random

import time

start_time = time.time()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

# combine with TIM method
if(FLAGS.combine_tim):
    kernel = gkern(11, 3).astype(np.float32)
else:
    kernel = gkern(1, 1).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def inceptionv3_model(x, noise=None, var=None):
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            input_diversity(x), num_classes=1001, noise=noise, var=var,
            is_training=False, reuse=tf.AUTO_REUSE)

    return logits_v3, end_points_v3

def inceptionv4_model(x, noise=None, var=None):
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
            input_diversity(x), num_classes=1001, noise=noise, var=var,
            is_training=False, reuse=tf.AUTO_REUSE)
    return logits_v4, end_points_v4


def inceptionresnetv2_model(x, noise=None, var=None):
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
            input_diversity(x), num_classes=1001,  noise=noise, var=var,
            is_training=False, reuse=tf.AUTO_REUSE)
    return logits_res_v2, end_points_res_v2


def resnet50_model(x, noise=None, var=None):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet, end_points_resnet = resnet_v2.resnet_v2_50(
            input_diversity(x), num_classes=1001, noise=noise, var=var,
            is_training=False, reuse=tf.AUTO_REUSE)
    return logits_resnet, end_points_resnet


def loss_function(endpoints1, y, is_auxnet=False):
    logits1 = endpoints1['Logits']
    loss = tf.losses.softmax_cross_entropy(y, logits1, label_smoothing=0.0, weights=1.0)
    if not is_auxnet:
        # adopt auxlogits for DIM and TIM
        if source_model != 'r50' and FLAGS.combine_sim != True:
            if(FLAGS.combine_dim or FLAGS.combine_tim):
                auxlogits = endpoints1['AuxLogits']
                loss += tf.losses.softmax_cross_entropy(y,auxlogits, weights=0.4)
    return loss

def data_erosion(x, x_initial):
    para = 1 - FLAGS.erosion_prob
    x_ = tf.tile(x, [FLAGS.dropout_num, 1, 1, 1])
    x_ = tf.divide(tf.nn.dropout(x_, keep_prob=para), tf.divide(1.0, para))
    x_dropout = tf.concat([x, x_], axis=0)
    x_init = tf.tile(x_initial, [FLAGS.dropout_num + 1, 1, 1, 1])
    return x_dropout, x_init


def get_endpoints(x, source_model, noise, var):
    if (source_model == 'i3'):
        _, end_points = inceptionv3_model(x, noise, var)
    elif (source_model == 'i4'):
        _, end_points = inceptionv4_model(x, noise, var)
    elif (source_model == 'ir2'):
        _, end_points = inceptionresnetv2_model(x, noise, var)
    elif (source_model == 'r50'):
        _, end_points = resnet50_model(x, noise, var)
    return end_points


def graph(x, x_initial, y, i, x_max, x_min, grad):


    if (FLAGS.max_epsilon > 1.0):
        eps = 2 * FLAGS.max_epsilon / 255.0
    else:
        eps = 2 * FLAGS.max_epsilon

    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum

    x_nes, x_init = data_erosion(x, x_initial)
    end_points = get_endpoints(x_nes, source_model, noise=None, var=None)

    pred = tf.argmax(end_points['Logits'], 1)
    pred = tf.tile(pred[0:FLAGS.batch_size], [FLAGS.dropout_num + 1])
    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    y = first_round * pred + (1 - first_round) * y
    one_hot = tf.one_hot(y, 1001)

    loss = loss_function(end_points, one_hot)
    noise = tf.gradients(loss, x)[0]

    # combine with SIM method
    if FLAGS.combine_sim:
        x_nes_4 = 1 / 4 * x_nes
        end_points_4 = get_endpoints(x_nes_4, source_model, noise=None, var=None)
        loss_4 = loss_function(end_points_4, one_hot)
        noise += tf.gradients(loss_4, x)[0]

        x_nes_8 = 1 / 8 * x_nes
        end_points_8 = get_endpoints(x_nes_8, source_model, noise=None, var=None)
        loss_8 = loss_function(end_points_8, one_hot)
        noise += tf.gradients(loss_8, x)[0]

    kernel_3 = gkern(3, 3).astype(np.float32)
    kernel_3 = tf.expand_dims(kernel_3, 0)

    # network_erosion
    end_points_ = get_endpoints(x_nes, source_model, noise=FLAGS.scale, var=kernel_3)
    aux_ep = get_aux_endpoints(end_points_, source_model)
    loss_aux = loss_function(aux_ep, one_hot, is_auxnet=True)
    noise += tf.gradients(loss_aux, x)[0]

    # combine with SIM method
    if FLAGS.combine_sim:
        end_points_4_ = get_endpoints(x_nes_4, source_model, noise=FLAGS.scale, var=kernel_3)
        aux_ep_4 = get_aux_endpoints(end_points_4_, source_model)
        loss_aux_4 = loss_function(aux_ep_4, one_hot, is_auxnet=True)
        noise += tf.gradients(loss_aux_4, x)[0]

        end_points_8_ = get_endpoints(x_nes_8, source_model, noise=FLAGS.scale, var=kernel_3)
        aux_ep_8 = get_aux_endpoints(end_points_8_, source_model)
        loss_aux_8 = loss_function(aux_ep_8, one_hot, is_auxnet=True)
        noise += tf.gradients(loss_aux_8, x)[0]

    noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise

    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)

    return x, x_initial, y, i, x_max, x_min, noise

def stop(x, x_initial, y, i, x_max, x_min, grad):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)


def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # combine with DIM method
    if(FLAGS.combine_dim):
        return ret
    else:
        return input_tensor

def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    # f2l = load_labels('./dev_data/val_rs.csv')
    if (FLAGS.max_epsilon > 1.0):
        eps = 2 * FLAGS.max_epsilon / 255.0
    else:
        eps = 2 * FLAGS.max_epsilon

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    check_or_create_dir(FLAGS.output_dir)

    print(time.time() - start_time)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_initial = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        y = tf.constant(np.zeros([FLAGS.batch_size * (FLAGS.dropout_num + 1)]), tf.int64)

        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)

        x_adv, _, _, _, _, _, _ = \
            tf.while_loop(stop, graph, [x_input, x_initial, y, i, x_max, x_min, grad])

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Run computation
            load_model(sess, source_model)
            load_aux_model(sess, source_model)

            output_dir = get_output_dir(source_model)
            print(output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print("create filefolder:", output_dir)

            idx = 0
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                idx += len(filenames)
                print("attack i={} images".format(idx), 'output_dir:', output_dir)
                adv_images = sess.run(x_adv, feed_dict={x_input: images,
                                                        x_initial: images})
                save_images(adv_images, filenames, output_dir)

            print(time.time() - start_time)

            print(output_dir)


if __name__ == '__main__':
    tf.app.run()
