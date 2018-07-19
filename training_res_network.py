from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
from collections import namedtuple
import tensorflow as tf
import resnet_model
import cifar_input

cifar10_data_path = '/Users/prmeasure/Desktop/cifar10/cifar-10-batches-bin'

def get_filenames(is_training, data_dir):
	assert os.path.exists(data_dir)
	if is_training:
		return [os.path.join(data_dir, 'data_batch_%d.bin' % i)
		        for i in range(1, 6)]
	else:
		return [os.path.join(data_dir, 'test_batch_%d.bin')]

Config = namedtuple('Config', ['batch_size', # the number of examples processed in each training batch.
                               'num_images', # 50000 for cifar10.
                               'batch_denom', # this value will be used to scale the base learning rate.
                               'boundary_epochs', # list of ints representing the epochs at
                                                # which we decay the learning rate.
                               'decay_rates', # list of floats representing the decay rates to be used
                               ])

class training_resnet(resnet_model.Model):
	def __init__(self, resnet_size, data_format,
	             num_classes, resnet_version,
	             dtype, samples_config, weight_decay, loss_scale=1):
		''' A class for training resnet.
		Args:
			resnet_size: The resnet model size.
			batch_denom:
			data_format: inputs data format, channel first or not.
			boundary_epochs:
			decies:
		'''
		if resnet_size % 6 != 2:
			raise ValueError('resnet size must be 6n + 2', resnet_size)
		num_blocks = int((resnet_size - 2) / 6)

		super(training_resnet, self).__init__(
			resnet_size=resnet_size,
			bottleneck=False,
			num_classes=num_classes,
			num_filters=16,
			kernel_size=3,
			conv_stride=1,
			first_pool_size=None,
			first_pool_stride=None,
			block_sizes=[num_blocks] * 3,
			block_strides=[1, 2, 2],
			final_size=64,
			resnet_version=resnet_version,
			data_format=data_format,
			dtype=dtype
		)

		self._samples_config = samples_config
		self._weight_decay = weight_decay
		self._learning_rate_fn = self._learning_rate_with_decay()
		self._loss_scale = loss_scale


	def _learning_rate_with_decay(self):
		initial_learning_rate = 0.1 * self._samples_config.batch_size / self._samples_config.batch_denom
		batchs_per_epoch = self._samples_config.num_images / self._samples_config.batch_size

		boundaris = []
		vals = []
		for epoch in self._samples_config.boundary_epochs:
			boundary = int(epoch * batchs_per_epoch)
			boundaris.append(boundary)

		for decay in self._samples_config.decay_rates:
			val = initial_learning_rate * decay
			vals.append(val)

		def learning_rate_fn(global_step):
			global_step = tf.cast(global_step, tf.int32)
			return tf.train.piecewise_constant(global_step, boundaris, vals)

		return learning_rate_fn

	def _train_resnet_model(self):
		''' Shared functionality for defferent resnet model_fn.'''
		data_path = get_filenames(is_training=True, data_dir=cifar10_data_path)
		features, labels = cifar_input.build_input(dataset='cifar10',
		                                           data_path=data_path,
		                                           batch_size=32,
		                                           mode='train')

		tf.summary.image('images', features, max_outputs=6)
		features = tf.cast(features, dtype=self.dtype)
		logits = self.network(features, training=True)
		logits = tf.cast(logits, tf.float32)

		predictions = {
			'classes': tf.argmax(logits, axis=1),
			'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
		}

		cross_entropy = tf.losses.sparse_softmax_cross_entropy(
			logits=logits, labels=labels)

		# Create a tensor named cross_entropy for logging purposes.
		tf.identity(cross_entropy, name='cross_entropy')
		tf.summary.scalar('cross_entropy', cross_entropy)

		# Add weight decay to the loss.
		l2_loss = self._weight_decay * tf.add_n(
			[tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])

		tf.summary.scalar('l2_loss', l2_loss)

		total_loss = cross_entropy + l2_loss
		global_step = tf.train.get_or_create_global_step()
		learning_rate = self._learning_rate_fn(global_step)
		tf.identity(learning_rate, name='learning_rate')
		tf.summary.scalar('learning_rate', learning_rate)

		optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
		                                       momentum=0.9)
		# TODO: add a loss_scale.
		if self._loss_scale != 1:
			scaled_grad_vars = optimizer.compute_gradients(total_loss * self._loss_scale)
			unscaled_grad_vars = [(grad / self._loss_scale, var)
			                      for grad, var in scaled_grad_vars]
			minimize_op = optimizer.apply_gradients(unscaled_grad_vars)
		else:
			minimize_op = optimizer.minimize(total_loss, global_step)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		train_op = tf.group(minimize_op, update_ops)

		# accuracy = tf.metrics.accuracy(labels, predictions['classes'])
		# metrics = {'accuracy': accuracy}
		# tf.identity(accuracy[1], name='train_accuracy')
		# tf.summary.scalar('train_accuracy', accuracy[1])

		return train_op, total_loss, predictions, global_step

	def training(self):
		# training loop
		# with tf.Graph().as_default():
		train_op, total_loss, predictions, global_step = self._train_resnet_model()
		# param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
		# 	tf.get_default_graph(),
		# 	tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
		# sys.stdout.write('Total parameters: %d\n' % param_stats.total_parameters)
		#
		# tf.contrib.tfprof.model_analyzer.print_model_analysis(
		# 	tf.get_default_graph(),
		# 	tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

		logging_hook = tf.train.LoggingTensorHook(
			tensors={'step': global_step,
			         'total_loss': total_loss,},
			every_n_secs=10)

		summary_hook = tf.train.SummarySaverHook(
			save_steps=100,
			output_dir='model_dir',
			summary_op=tf.summary.merge_all())

		with tf.train.MonitoredTrainingSession(
			checkpoint_dir='checkpoints',
			hooks=[logging_hook],
			chief_only_hooks=[summary_hook],
			save_summaries_steps=100,
			config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			while not sess.should_stop():
				sess.run(train_op)


if __name__ == '__main__':
	learning_rate_config = Config(batch_size=32,
	                              num_images=50000,
	                              batch_denom=128,
	                              boundary_epochs=[100, 150, 200],
	                              decay_rates=[1, 0.1, 0.01, 0.001],
	                              )

	model = training_resnet(resnet_size=50,
	                        data_format=None,
	                        num_classes=10,
	                        resnet_version=2,
	                        dtype=tf.float32,
	                        samples_config=learning_rate_config,
	                        weight_decay=0.0002,
	                        loss_scale=1)
	model.training()






















