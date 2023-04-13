import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from nets import inception_resnet_v2_modify as inception_resnet_v2
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

INCEPTION_CHECKPOINT_PATH = '/home/zhuangwz/dataset/ckpt_models/inception_resnet_v2_2016_08_30.ckpt'
input_path = '/home/zhuangwz/dataset/ILSVRC2012/val'
output_path = '/home/zhuangwz/code/sa_result/train_model/'
batch_size = 16
epoch_num = 10
model = 'Aux_InceptionResnetV2'
output_path = os.path.join(output_path, model)
out_layer = ['block17']
# out_layer = ['Mixed_5b','block35','Mixed_6a','block17','Mixed_7a','block8','Conv2d_7b_1x1']
filepath = tf.gfile.Glob(os.path.join(input_path, '*.JPEG'))

def get_image(batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    random.shuffle(filepath)
    for f in filepath:
        image = imresize(imread(f, mode='RGB'), [299, 299]).astype(np.float) / 255.0
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(f))
        idx += 1
        if idx == batch_size:
            return images, filenames
    return images, filenames

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
    for filepath in input_dir:
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imresize(imread(f, mode='RGB'), [299, 299]).astype(np.float) / 255.0
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


def inceptionresnetv2_model(x, noise=None, var=None):
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_v3, end_points_v3 = inception_resnet_v2.inception_resnet_v2(
            x, num_classes=1001, noise=noise, var=var,
            is_training=False, reuse=tf.AUTO_REUSE)
    return logits_v3, end_points_v3

def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.

  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                         tf.minimum(shape[2], kernel_size[1])])

  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


num_classes =1001
tf.reset_default_graph()
x = tf.placeholder(dtype='float32', shape=[batch_size, 299, 299, 3], name='input_image')
logits, endpoints = inceptionresnetv2_model(x)
predicted_labels = tf.argmax(endpoints['Predictions'], 1)
y = tf.one_hot(predicted_labels, num_classes)

import scipy.stats as st
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel
kernel_3 = gkern(3, 3).astype(np.float32)
kernel_3 = tf.expand_dims(kernel_3, 0)
# er_logits, er_endpoints = inceptionresnetv2_model(x, noise=0.1, var=kernel_3)
er_logits, er_endpoints = inceptionresnetv2_model(x)

def loss_func(aux_logits, y):
    ce = tf.losses.softmax_cross_entropy(y, aux_logits)
    return ce

opt = tf.train.AdamOptimizer(learning_rate=1e-3)

train_num = len(out_layer)
logits_list = []
var_list = []
loss_list = []
op_list = []
initial_list = []
scope_list = []
dropout_keep_prob = 0.8
for i in range(train_num):
    scope_name = model + '_' + out_layer[i]
    scope_list.append(scope_name)
    with tf.variable_scope(scope_name):
        # net = endpoints[out_layer[i]]
        net = er_endpoints[out_layer[i]]
        size = net.shape[2]
        kernel_size = _reduced_kernel_size_for_small_input(net, [size, size])
        net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                              scope='AvgPool_1a_{}x{}'.format(*kernel_size))
        net = slim.dropout(net, dropout_keep_prob, scope='Dropout')
        aux_logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
        logits_list.append(aux_logits)
        loss_list.append(loss_func(aux_logits, y))
    loss = loss_func(aux_logits, y)
    train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
    var_list.append(train_var)
    grads = opt.compute_gradients(loss, train_var)
    op_list.append(opt.apply_gradients(grads))
    t_vars = tf.trainable_variables(scope=scope_name)
    var_init = tf.variables_initializer(t_vars)
    initial_list.append(var_init)

s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
save_list = []
for i in range(train_num):
    s2 = tf.train.Saver(slim.get_model_variables(scope=scope_list[i]), max_to_keep = 20)
    save_list.append(s2)

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    var = tf.global_variables()
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.variables_initializer(var_list=train_var))
    s1.restore(sess, INCEPTION_CHECKPOINT_PATH)
    print('load ckpt:', INCEPTION_CHECKPOINT_PATH)
    for i in range(train_num):
        sess.run(initial_list[i])

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("create filefolder:", output_path)

    batch_shape = [batch_size, 299, 299, 3]
    for epoch in range(epoch_num):
        cur_idx = 0
        random.shuffle(filepath)
        for filenames, images in load_images(filepath, batch_shape):
            if (cur_idx % 500 == 0):
                _, ce_np = sess.run([op_list, loss_list], feed_dict={x: images})
                print('epoch:', epoch + 1, 'idx:', cur_idx, 'ce:', ce_np)
            if (cur_idx % 2000 == 0):
                for i in range(train_num):
                    ckpt_name = scope_list[i] + '.ckpt'
                    path = os.path.join(output_path, ckpt_name)
                    save_list[i].save(sess, path, global_step=epoch + 1)
            sess.run(op_list, feed_dict={x: images})
            cur_idx += 1
