import tensorflow as tf
import tensorflow.contrib.slim as slim
import distortimg
import os

# define the flags
flags = tf.app.flags

#flags.DEFINE_integer('epochs', 10, 'input directory of training data')
#flags.DEFINE_string('logdir', '/home/admin/PycharmProjects/facex/logdir', 'checkpoint directory')
FLAGS = flags.FLAGS

def _read_input_tfr(filename_queue,tfrimg,tfrlabel):
    '''
    read tfrecords file. default inputshape of image is [height, wideth, channel]
    below flags need be set:channel/orgheight/orgwideth
    :param filename_queue:
    :param tfrimg:
    :param tfrlabel:
    :return:
    '''
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
        features={tfrimg:tf.FixedLenFeature([], tf.string), tfrlabel:tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(features[tfrimg], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [FLAGS.orgheight,FLAGS.orgwideth,FLAGS.channel], name = tfrimg)
    label = tf.cast(features[tfrlabel], tf.int32, name = tfrlabel)
    label.set_shape([])
    label = tf.reshape(label, [1])
    return image, label

def _read_input_bin(filename_queue,label_len=1):
    '''
    Read bin file. default inputshape of image is [channel, height, wideth],
    if it is not so, you need create a new method.
    Below flags need be set:channel/orgheight/orgwideth
    :param filename_queue:
    :param label_len:
    :param height:
    :param wideth:
    :return:
    '''
    image_len = FLAGS.channel * FLAGS.orgheight * FLAGS.orgwideth
    record_len = image_len + label_len
    reader = tf.FixedLengthRecordReader(record_bytes=record_len)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    label = tf.cast(tf.strided_slice(record_bytes, [0], [label_len]), tf.int32)
    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    image = tf.reshape(
        tf.strided_slice(record_bytes, [label_len],
                         [record_len]),[FLAGS.channel, FLAGS.orgheight, FLAGS.orgwideth])
    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.transpose(image, [1, 2, 0])
    image = tf.cast(image, tf.float32)
    #the shape of label is unkown after tf.strided_slice op, we need set the shape.
    label.set_shape([1])

    return image, label

def input(input_dir, batch_size = 128, num_threads = 16, format = 'tfr',
          tfrimg='img_raw',tfrlabel = 'label', one_hot=False,numofclass=10, distort = True,shuffle=True):
    '''
    read records using filename queue and get batch examples
    Below flags need be set:newheight/newwide
    :param input_dir:
    :param batch_size:
    :param num_threads:
    :param format:
    :param tfrimg:
    :param tfrlabel:
    :param one_hot:
    :param numofclass:
    :param distort:
    :param shuffle:
    :return:
    '''
    filenames = [input_dir+file for file in os.listdir(input_dir)]
    filename_queue = tf.train.string_input_producer(filenames)
    if format=='tfr':
        image, label = _read_input_tfr(filename_queue,tfrimg,tfrlabel)
    elif format == 'bin':
        image, label = _read_input_bin(filename_queue)
    else:
        'error: invalid file format!'

    if distort:
        image = distortimg.distort(image, FLAGS.newheight, FLAGS.newwideth)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.newheight, FLAGS.newwideth)
        image = tf.image.per_image_standardization(image)

    if one_hot:
        label = tf.one_hot(label, numofclass, dtype = tf.int32)


    if shuffle:
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size = batch_size, num_threads = num_threads,
            capacity = FLAGS.queue_capacity+batch_size*5, min_after_dequeue =FLAGS.queue_capacity)
    else:
        images, sparse_labels = tf.train.batch(
            [image, label], batch_size=batch_size, num_threads=num_threads,
            capacity=FLAGS.queue_capacity+batch_size*5)
    if one_hot:
        sparse_labels = tf.reshape(sparse_labels, [batch_size, numofclass])
    else:
        sparse_labels = tf.reshape(sparse_labels, [batch_size])
    return images, tf.cast(sparse_labels, tf.float32)


