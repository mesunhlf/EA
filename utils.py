import tensorflow as tf
import os
from nets import inception_resnet_v2_modify as inception_resnet_v2
from nets import resnet_v2_modify as resnet_v2
from nets import inception_v3_modify as inception_v3
from nets import inception_v4_modify as inception_v4
import numpy as np

slim = tf.contrib.slim

tf.flags.DEFINE_integer('gpu_num', 1, 'How many images process at one time.')

tf.flags.DEFINE_integer('batch_size', 8, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 0.05, 'max epsilon.')

tf.flags.DEFINE_integer('num_iter', 10, 'max iteration.')

tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_float('prob', 0.7, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer('image_resize', 331, 'Height of each input images.')

tf.flags.DEFINE_integer('dropout_num', 3, 'Number of data erosion images.')

tf.flags.DEFINE_float('erosion_prob', 0.1, 'Parameter of data erosion.')

tf.flags.DEFINE_float('scale', 0.02, 'Parameter of network erosion.')

tf.flags.DEFINE_boolean('combine_dim', False, 'use dim method or not.')

tf.flags.DEFINE_boolean('combine_tim', False, 'use tim method or not.')

tf.flags.DEFINE_boolean('combine_sim', False, 'use sim method or not.')

tf.flags.DEFINE_string('checkpoint_path', '/home/zhuangwz/code/ckpt_models/',
                       'Path to checkpoint for pretained models.')

tf.flags.DEFINE_string('aux_checkpoint_path', '/home/zhuangwz/code/erosion_model/',
                       'Path to checkpoint for erosion models.')

tf.flags.DEFINE_string('input_dir', '/home/zhuangwz/dataset/ali2019/images1000_val/attack',
                       'Input directory with images.')

tf.flags.DEFINE_string('output_dir',
                       '/home/zhuangwz/code/sa_result/result/',
                       'Output directory with images.')

FLAGS = tf.flags.FLAGS
num_classes = 1001

source_model = 'i4' # i3 , i4, ir2, r50
ensemble_models = ['i3', 'i4', 'ir2', 'r50']
epoch = '10'

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'resnet_v2_50': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_50.ckpt'),
}

aux_pre_scope_map = {
    'i3' : 'Aux_InceptionV3',
    'i4' : 'Aux_InceptionV4',
    'ir2' : 'Aux_InceptionResnetV2',
    'r50' : 'Aux_resnet_v2_50'
}

aux_post_scope_map = {
    'i3': 'Mixed_6e',
    'i4': 'Mixed_6g',
    'ir2': 'block17',
    'r50': 'resnet_v2_50/block3/unit_6/bottleneck_v2'
}


def get_output_dir(source_model):
    aux = 'erosion-attack'
    output_dir = os.path.join(FLAGS.output_dir, aux, source_model, 'attack')
    return output_dir


def load_model(sess, source_model):
    if (source_model == 'i3'):
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s1.restore(sess, model_checkpoint_map['inception_v3'])
    elif (source_model == 'i4'):
        s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s2.restore(sess, model_checkpoint_map['inception_v4'])
    elif (source_model == 'r50'):
        s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
        s4.restore(sess, model_checkpoint_map['resnet_v2_50'])
    elif (source_model == 'ir2'):
        s4 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s4.restore(sess, model_checkpoint_map['inception_resnet_v2'])
    elif (source_model == 'ensemble'):
        for i in range(len(ensemble_models)):
            model = ensemble_models[i]
            load_model(sess, model)

def get_aux_scope_name(source_model):
    if(source_model == 'ensemble'):
        pre_scope_name = {}
        post_scope_name = {}
        scope_name = {}
        for i in range(len(ensemble_models)):
            model = ensemble_models[i]
            pre_scope_name_, post_scope_name_, scope_name_ = get_aux_scope_name(model)
            pre_scope_name[model] = pre_scope_name_
            post_scope_name[model] = post_scope_name_
            scope_name[model] = scope_name_
    else:
        pre_scope_name = aux_pre_scope_map[source_model]
        post_scope_name = aux_post_scope_map[source_model]
        post_scope_name_rename = str(np.char.replace(post_scope_name, '/', '_'))
        scope_name = pre_scope_name + '_' + post_scope_name_rename

    return pre_scope_name, post_scope_name, scope_name

def load_aux_model(sess, source_model):
    if(source_model == 'ensemble'):
        for i in range(len(ensemble_models)):
            model = ensemble_models[i]
            load_aux_model(sess, model)
    else:
        pre_scope_name, _, scope_name = get_aux_scope_name(source_model)
        aux_checkpoint_path = os.path.join(FLAGS.aux_checkpoint_path, pre_scope_name)
        aux_checkpoint_name = scope_name + '.ckpt-' + epoch
        aux_checkpoint_path = os.path.join(aux_checkpoint_path, aux_checkpoint_name)
        print(aux_checkpoint_path)
        s_aux = tf.train.Saver(slim.get_model_variables(scope=scope_name))
        s_aux.restore(sess, aux_checkpoint_path)

def get_aux_endpoints(endpoints, source_model):
    aux_endpoints = {}
    pre_scope_name, post_scope_name, scope_name = get_aux_scope_name(source_model)

    if(source_model == 'i3' or source_model == 'ir2' or source_model == 'i4'):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            net = endpoints[post_scope_name]
            size = net.shape[1]
            kernel_size = inception_v3._reduced_kernel_size_for_small_input(net, [size, size])
            net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                  scope='AvgPool_1a_{}x{}'.format(*kernel_size))
            aux_logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1')
            aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
            aux_endpoints['Logits'] = aux_logits

    elif(source_model == 'r50'):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            net = endpoints[post_scope_name]
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                              normalizer_fn=None, scope='logits')
            aux_logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            aux_endpoints['Logits'] = aux_logits

    elif(source_model == 'ensemble'):
        aux_endpoints = {}
        for i in range(len(ensemble_models)):
            model = ensemble_models[i]
            aux_endpoint_single = get_aux_endpoints(endpoints[model], model)
            aux_endpoints[model] = aux_endpoint_single
    return aux_endpoints


def cos_value(x1, x2):
    x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1)))
    x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2)))
    x1_x2 = tf.reduce_sum(tf.multiply(x1, x2))
    cosin = x1_x2 / (x1_norm * x2_norm)
    return cosin
