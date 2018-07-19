def func_1(_):
	return True

def exclude_batch_norm (name):
		return 'batch_normalization' not in name


funcc = False or False

from collections import namedtuple

Config = namedtuple('Config', ['batch_size',
                                       'batch_denom',
                                       'boundary_epochs',
                                       'decies'])
sample_config = Config(batch_size=32,
                       batch_denom=1,
                       boundary_epochs=10,
                       decies=10)

print(sample_config.batch_size)
