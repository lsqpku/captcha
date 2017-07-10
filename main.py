'''
this is basic tensorflow train+eval mode. include below function point:
1) save and restore checkpoint file.
2) summary function.
be noted that this is for 4 digit yanzhengma example, which concatate 4 sub-layers into
a big layer.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import alexnet
import tensorflow.models as models
    #.slim.nets.alexnet as alexnet
import liuinput
import liumodel
import numpy as np
import cv2
# Basic model parameters as external flags.
flags = tf.app.flags
# input dir: only training data, all will be read for training.
flags.DEFINE_string('train_dir', 'train/', 'input directory of training data')
flags.DEFINE_string('validation_dir', 'validation/', 'input directory of validation data')
flags.DEFINE_string('test_dir', 'test/', 'input directory of test data')
flags.DEFINE_string('checkpointdir', 'checkpoint_yanzhengma/', 'checkpoint directory')
flags.DEFINE_integer('epoches', 500000 , 'input directory of training data')
flags.DEFINE_integer('classes', 28, 'number of classes')
flags.DEFINE_integer('channel', 3, 'channels of image')
flags.DEFINE_integer('newheight', 224, 'croped image height')
flags.DEFINE_integer('newwideth', 224, 'croped image wideth')
flags.DEFINE_integer('orgheight', 256, 'original image height')
flags.DEFINE_integer('orgwideth', 256, 'original image wideth')
flags.DEFINE_integer('queue_capacity', 10000, 'capac    ity of file queue')
flags.DEFINE_float('loss_decay', 0.004, 'loss decay weight')
flags.DEFINE_float('lr', 0.1, 'learning rate')
flags.DEFINE_integer('batch_size',128, 'batch size of examples')
FLAGS = flags.FLAGS
import random
from captcha.image import ImageCaptcha
number = '0123456789'
def gen_num():
    id = ''
    for i in range(4):
        id += random.choice(number)
    return id

def input(batch, font):
    data=[]
    label=[]
    pic = np.ndarray((batch, 60, 160, 3), dtype=float)
    for i in xrange(batch):
        num = gen_num()
        img = font.generate(num)
        img = np.fromstring(img.getvalue(), dtype = 'uint8')
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        cv2.imwrite("tmp_yanzhengma/tmp"+str(i % 10)+".png",img)
        img = np.multiply(img, 1/255.0)
        data.append(img)
        label.append(int(num[0]))
        for y in range(60):
            for z in range(160):
                for m in range(3):
                    pic[i, y, z, m] = img[y,z,m]

        label.append(int(num[1]))
        label.append(int(num[2]))
        label.append(int(num[3]))
    newlabel = []
    for i in xrange(batch):
        newlabel.append(label[i*4])
    for i in xrange(batch):
        newlabel.append(label[i*4+1])
    for i in xrange(batch):
        newlabel.append(label[i*4 + 2])
    for i in xrange(batch):
        newlabel.append(label[i*4+3])
    return pic, newlabel


def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,60,160,3))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size*4))
  return images_placeholder, labels_placeholder

def fill_feed_dict(images_pl, labels_pl, batch_size,font):
  images_feed, labels_feed = input(batch_size, font)
  feed_dict = {images_pl: images_feed,labels_pl: labels_feed}
  return feed_dict

def do_eval(sess,eval_correct,images_placeholder,labels_placeholder,batch_size, font,sumevalprc_op):
  feed_dict = fill_feed_dict(images_placeholder,labels_placeholder,batch_size, font)
  true_count = sess.run(eval_correct, feed_dict=feed_dict)
  summary_str = sess.run(sumevalprc_op, feed_dict=feed_dict)
  precision = float(true_count) / (batch_size*4)
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (batch_size*4, true_count, precision))
  return summary_str

def run_training():
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
    # Build a Graph that computes predictions from the inference model.
    predict = liumodel.yanzhengma(images_placeholder)

    # Add to the Graph the Ops for loss calculation.
    tf.losses.sparse_softmax_cross_entropy(labels_placeholder, predict)
    total_loss = tf.losses.get_total_loss()
    sumloss_op = tf.summary.scalar('liuloss', total_loss)
    # Add to the Graph the Ops that calculate and apply gradients.
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.lr)
    train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=False)
    # Add the Op to compare the logits to the labels during evaluation.
    correct = tf.nn.in_top_k(predict, labels_placeholder, 1)
    eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
    eval_sum = tf.divide(eval_correct, FLAGS.batch_size*4)
    sumevalprc_op = tf.summary.scalar('evalprecision', eval_sum)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    # Add the variable initializer Op.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    font = ImageCaptcha(fonts=['tmp_yanzhengma/font.ttf'], height=60, width=160)
    # Create a session for running Ops on the Graph.
    with tf.Session() as sess:
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.checkpointdir, sess.graph)
        # Run the Op to initialize the variables.
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpointdir)
        start_step = 0
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint_file=ckpt.model_checkpoint_path
            print(checkpoint_file)
            start_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            start_step = int(start_step)
            saver.restore(sess,checkpoint_file)
        # Start the training loop.
        for step in xrange(start_step, FLAGS.epoches):
          start_time = time.time()

          # Fill a feed dictionary with the actual set of images and labels
          # for this particular training step.
          feed_dict = fill_feed_dict(images_placeholder,labels_placeholder,
                                     FLAGS.batch_size,font
                                     )
          _, loss_value = sess.run([train_op, total_loss],
                                   feed_dict=feed_dict)

          duration = time.time() - start_time

          # Write the summaries and print an overview fairly often.
          if step % 10 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            # Update the events file.
            summary_str = sess.run(sumloss_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

          # Save a checkpoint and evaluate the model periodically.
          if (step + 1) % 10 == 0 or (step + 1) == FLAGS.epoches:
            checkpoint_file = os.path.join(FLAGS.checkpointdir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step)

            # Evaluate against the test set.
            print('Validation Data Eval:')
            feed_dict = fill_feed_dict(images_placeholder, labels_placeholder, FLAGS.batch_size, font)
            true_count = sess.run(eval_correct, feed_dict=feed_dict)
            precision = float(true_count) / (FLAGS.batch_size * 4)
            print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                  (FLAGS.batch_size * 4, true_count, precision))
            summary_str = sess.run(sumevalprc_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
            predict_result = sess.run(predict, feed_dict=feed_dict)
            print('predict is:')
            print(predict_result)
            '''
            for xx in xrange(FLAGS.batch_size*4):
                print('predict: ', np.argmax(predict_result[xx]))
                print('target:',feed_dict[labels_placeholder][xx] )


            predict_result = sess.run(predict, feed_dict=feed_dict)
            # print('predict is:')
            # print(predict_result)
            label_result = feed_dict[labels_placeholder]
            # print('target is:')
            # print(label_result)

            cnt = 0
            for item in xrange(FLAGS.batch_size*4):
                if predict_result[item, label_result[item] ]== label_result[item]:
                   cnt = cnt + 1
            print(cnt/(FLAGS.batch_size*4))
            '''


def main():
  run_training()

if __name__ == '__main__':
    main()
