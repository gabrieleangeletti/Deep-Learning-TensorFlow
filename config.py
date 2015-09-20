# Datasets sizes
TRAIN_SET_SIZE = 30000
TEST_SET_SIZE = 10000

####################################################################
# Logistic Regression Classifier configuration
# EMPTY
####################################################################
# Standard RBM configuration
NV = 784 			# number of visible units
NH = 500    		# number of hidden units
# Train parameters
EPOCHS = 500        # number of training epochs
ALPHA = 0.01		# learning rate
M = 0.0				# momentum parameter
BATCH_SIZE = 100     # size of each batch
GIBBS_K = 1			# number of gibbs sampling steps
ALPHA_UPDATE_RULE = 'constant'  # type of update rule for the learning rate
VERBOSE = True  # if true, a progress bar and a reconstructed sample for each epoch will be shown
OUTFILE = 'models/rbm.json'  # outfile to save the configuration of the rbm after training
# Mind reader
FANTASY_K = 1000        # number of gibbs sampling steps for reading the mind of the rbm
# Weights image
HS = 100  # how many hidden unit weights to save
WIDTH = 28
HEIGHT = 28
W_OUTFILE = 'models/rbm_w.png'  # outfile for the hidden weights image/s
####################################################################
# Multinomial RBM configuration
MULTI_NV = 784  # number of visible units
MULTI_NH = 150  # number of hidden units
MULTI_KV = 1  # visible units can be between 0 and this parameter
MULTI_KN = 1  # hidden units can be between 0 and this parameter
# Train parameters
M_EPOCHS = 200   # number of training epochs
M_ALPHA = 0.01		# learning rate
M_M = 0.0				# momentum parameter
M_BATCH_SIZE = 25     # size of each batch
M_GIBBS_K = 1			# number of gibbs sampling steps
M_ALPHA_UPDATE_RULE = 'constant'  # type of update rule for the learning rate
M_VERBOSE = True  # if true, a progress bar and a reconstructed sample for each epoch will be shown
M_OUTFILE = 'models/mrbm.json'  # outfile to save the configuration of the rbm after training
####################################################################
# Gaussian RBM configuration
GAUSS_NV = 784            # number of visible units
GAUSS_NH = 200            # number of hidden units
# Train parameters
G_EPOCHS = 100   # number of training epochs
G_ALPHA = 0.0001       # learning rate
G_M = 0.0             # momentum parameter
G_BATCH_SIZE = 10     # size of each batch
G_GIBBS_K = 1         # number of gibbs sampling steps
G_ALPHA_UPDATE_RULE = 'constant'  # type of update rule for the learning rate
G_VERBOSE = True  # if true, a progress bar and a reconstructed sample for each epoch will be shown
G_OUTFILE = 'models/grbm.json'  # outfile to save the configuration of the rbm after training
####################################################################
# Deep Belief Network configuration
DBN_LAYERS = [784, 500, 500]
DBN_LAST_LAYER = 2000
# Train parameters
DBN_EPOCHS = 150        # number of training epochs
DBN_ALPHA = 0.05		# learning rate
DBN_M = 0.0				# momentum parameter
DBN_BATCH_SIZE = 20     # size of each batch
DBN_GIBBS_K = 1			# number of gibbs sampling steps
DBN_ALPHA_UPDATE_RULE = 'constant'  # type of update rule for the learning rate
DBN_VERBOSE = True  # if true, a progress bar and a reconstructed sample for each epoch will be shown
DBN_OUTFILE = 'models/dbn.json'  # outfile to save the configuration of the rbm after training
DBN_LAST_LAYER_OUTFILE = 'models/last_layer_rbm.json'  # outfile to save the configuration of the rbm after training
####################################################################
