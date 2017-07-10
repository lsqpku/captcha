import tensorflow as tf
import tensorflow.contrib.slim as slim
import distortimg
# important note:
# this is collections of all models.
FLAGS = tf.app.flags.FLAGS
def model(image):
    with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.001)):
        net = slim.layers.conv2d(image, 64, [2,2],scope='covn1')
        net = slim.layers.conv2d(net, 128, [2, 2], scope='covn1-1')
        net = slim.layers.max_pool2d(net,  [2, 2], scope='pool1')
        net = slim.layers.conv2d(net, 128, [2, 2], scope='covn2')
        net = slim.layers.conv2d(net, 128, [2, 2], scope='covn2-1')
        net = slim.layers.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.layers.conv2d(net, 256, [2, 2], scope='covn2-3')
        net = slim.layers.conv2d(net, 256, [2, 2], scope='covn2-4')
        net = slim.layers.max_pool2d(net, [2, 2], scope='pool2-1')
        net = slim.layers.conv2d(net, 512, [2, 2], scope='covn3')
        net = slim.layers.conv2d(net, 512, [2, 2], scope='covn3-1')
        net = slim.layers.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.layers.flatten(net, scope='flatten1')
        net = slim.layers.fully_connected(net, 1024, scope='fc1')
        net = slim.layers.dropout(net, scope='dropout1')
        net = slim.layers.fully_connected(net, 4096, scope='fc3')
        net = slim.layers.dropout(net, scope='dropout1')
        net = slim.layers.fully_connected(net, 28, activation_fn=None, scope='fc2')
    return net

def model2(image):
    with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.001)
                        ):
        net = slim.layers.conv2d(image, 64, [4,4],scope='covn1')
        net = slim.layers.conv2d(net, 128, [4, 2], scope='covn1-1')
        net = slim.layers.max_pool2d(net,  [4, 2], scope='pool1')
        net = slim.layers.conv2d(net, 128, [4, 4], scope='covn2-1')
        net = slim.layers.max_pool2d(net, [4, 4], scope='pool2')
        net = slim.layers.conv2d(net, 256, [4, 4], scope='covn2-4')
        net = slim.layers.max_pool2d(net, [4, 4], scope='pool2-1')
        net = slim.layers.conv2d(net, 512, [4, 4], scope='covn3-1')
        net = slim.layers.max_pool2d(net, [4, 4], scope='pool4')
        net = slim.layers.flatten(net, scope='flatten1')
        net = slim.layers.fully_connected(net, 1024, scope='fc1')
        net = slim.layers.dropout(net, scope='dropout1')
        net = slim.layers.fully_connected(net, 4096, scope='fc3')
        net = slim.layers.dropout(net, scope='dropout1')
        net = slim.layers.fully_connected(net, 28, activation_fn=None, scope='fc2')
    return net


def vgg16(images, numofclass = 1000):
    '''
    this is vgg 16 model, the super params such as kernel size and stride are set and
    should not be changed.

    :param images: a batch of images [batchsize, height, wideth, channels]
    :param numofclass: number of classes
    :return: Tensor [batchsize, numofclass]
    '''
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.fully_connected(net, 4096, scope='fc6')
        net = slim.dropout(net, 0.5, scope='dropout6')
        net = slim.fully_connected(net, 4096, scope='fc7')
        net = slim.dropout(net, 0.5, scope='dropout7')
        net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
    return net

def loss_with_decay(var):

    return tf.multiply(FLAGS.loss_decay, tf.nn.l2_loss(var))

def alexnet1(images, numofclass=10):
    '''
    this model is extracted from official cifar10 tutorial
    and is similar with Alexnet. Be noted the loss weight decay of fc1&fc2 is not included here.
    :param images: a batch of images [batchsize, height, wideth, channels]
    :param numbofclass: number of classes
    :return: a Tensor [batchsize, numofclass]
    '''
    net = slim.layers.conv2d(images, 64, [5, 5], scope='covn1',weights_initializer=tf.truncated_normal_initializer(0.0, 0.05))
    net = slim.layers.max_pool2d(net, [3, 3], padding='SAME',scope='pool')
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
    net = slim.layers.conv2d(net, 64, [5, 5], scope='covn2',weights_initializer=tf.truncated_normal_initializer(0.0, 0.05), biases_initializer=tf.constant_initializer(0.1))
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
    net = slim.layers.max_pool2d(net, [3, 3], padding='SAME',scope='pool2')
    net = slim.layers.flatten(net, scope='flatten1')
    '''
    important note: weights_regularizer makes big difference.
    '''
    net = slim.layers.fully_connected(net, 384, scope='fc1',weights_initializer=tf.truncated_normal_initializer(0.0, 0.04), biases_initializer=tf.constant_initializer(0.1),weights_regularizer=loss_with_decay)
    net = slim.layers.fully_connected(net, 192, scope='fc2',weights_initializer=tf.truncated_normal_initializer(0.0, 0.04), biases_initializer=tf.constant_initializer(0.1),weights_regularizer=loss_with_decay)
    net = slim.layers.fully_connected(net, numofclass, activation_fn=None, scope='fc3',weights_initializer=tf.truncated_normal_initializer(0.0, 1 / 192.0))

    return net

def yanzhengma(images, numofclass=10):
    '''
    this model is extracted from official cifar10 tutorial
    and is similar with Alexnet. Be noted the loss weight decay of fc1&fc2 is not included here.
    :param images: a batch of images [batchsize, height, wideth, channels]
    :param numbofclass: number of classes
    :return: a Tensor [batchsize, numofclass]
    '''
    net = slim.layers.conv2d(images, 32, [5, 5], scope='covn1',weights_initializer=tf.truncated_normal_initializer(0.0, 0.05))
    net = slim.layers.max_pool2d(net, [2, 2], padding='SAME',scope='pool')
    #net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
    net = slim.layers.conv2d(net, 32, [5, 5], scope='covn2',weights_initializer=tf.truncated_normal_initializer(0.0, 0.05), biases_initializer=tf.constant_initializer(0.1))
    net = slim.layers.avg_pool2d(net, [2,2], padding = 'SAME', scope = 'pool1')
    #net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
    net = slim.layers.conv2d(net, 32, [3, 3], scope='covn3',
                             weights_initializer=tf.truncated_normal_initializer(0.0, 0.05),
                             biases_initializer=tf.constant_initializer(0.1))
    net = slim.layers.avg_pool2d(net, [2, 2], padding='SAME',scope='pool2')
    net = slim.layers.flatten(net, scope='flatten1')

    net = slim.layers.fully_connected(net, 512, scope='fc1',weights_initializer=tf.truncated_normal_initializer(0.0, 0.04), biases_initializer=tf.constant_initializer(0.1),weights_regularizer=loss_with_decay)
    net21 = slim.layers.fully_connected(net, 10, scope='fc21',weights_initializer=tf.truncated_normal_initializer(0.0, 0.04), biases_initializer=tf.constant_initializer(0.1),weights_regularizer=loss_with_decay)
    net22 = slim.layers.fully_connected(net, 10, scope='fc22',
                                       weights_initializer=tf.truncated_normal_initializer(0.0, 0.04),
                                       biases_initializer=tf.constant_initializer(0.1),
                                       weights_regularizer=loss_with_decay)
    net23 = slim.layers.fully_connected(net, 10, scope='fc23',
                                       weights_initializer=tf.truncated_normal_initializer(0.0, 0.04),
                                       biases_initializer=tf.constant_initializer(0.1),
                                       weights_regularizer=loss_with_decay)
    net24 = slim.layers.fully_connected(net, 10, scope='fc24',
                                       weights_initializer=tf.truncated_normal_initializer(0.0, 0.04),
                                       biases_initializer=tf.constant_initializer(0.1),
                                       weights_regularizer=loss_with_decay)
    net = tf.concat([net21,net22,net23,net24], 0)

    return net
