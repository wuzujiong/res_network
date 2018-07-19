import tensorflow as tf


training_config = {
	# model parameters define.
	'resnet_size': '50',
	'num_gpus': 1,
	'data_format': 'NWHC',
	'resnet_version': 2,
	# TODO: add the loss_scale.
	'loss_scale': '',
	'dtype': tf.float16,

	# Sample training config.
	'batch_size': 32,
	'train_epochs': 100,
	# TODO: add the model dir.
	'model_dir': '',
	'hooks': None,

	# config device.
	'inter_op_parallelism_threads': 4,
	'intra_op_parallelism_threads': 4,

}