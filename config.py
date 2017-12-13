import ConfigParser

config = ConfigParser.RawConfigParser()

config.add_section('Parameter')
config.set('Parameter', 'batch_size', '128')
config.set('Parameter', 'learning_rate', '0.001')
config.set('Parameter', 'iteration', '3000')
config.set('Parameter', 'hidden_dim', '8')
config.set('Parameter', 'test_ratio', '0.2')
config.set('Parameter', 'train_dir', 'log/train')
config.set('Parameter', 'l2_regularizer_use', 0)
config.set('Parameter', 'what_data_use', 'both')
config.set('Parameter', 'save_result_dir', 'result2')

config.set('Parameter', 'data_dir_beta', '~')
config.set('Parameter', 'data_dir_rna', '~')
config.set('Parameter', 'data_dir_both', '~')

with open('config.cfg', 'wb') as configfile:
	config.write(configfile)
