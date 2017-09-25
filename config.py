import ConfigParser

config = ConfigParser.RawConfigParser()

config.add_section('Parameter')
config.set('Parameter', 'batch_size', '80')
config.set('Parameter', 'learning_rate', '0.1')
config.set('Parameter', 'iteration', '1000')
config.set('Parameter', 'hidden_dim', '32')

config.set('Parameter', 'train_dir', 'log/train')

#config.set('Parameter', 'data_dir_beta', '../data/ADD_cor_and_beta_0.15.site.ham21.ham17.ssi')
config.set('Parameter', 'data_dir_beta', '../data/ADD_cor_and_beta_0.35.site.ham21.ham17.ssi')

#config.set('Parameter', 'data_dir_rna', '../data/ADD_cor_and_rna_0.15.site.ham21.ham17.ssi')
config.set('Parameter', 'data_dir_rna', '../data/ADD_cor_and_rna_0.35.site.ham21.ham17.ssi')

with open('config.cfg', 'wb') as configfile:
	config.write(configfile)
