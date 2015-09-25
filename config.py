# Datasets sizes
TRAIN_SET_SIZE = 1000
TEST_SET_SIZE = 1000

####################################################################
# Logistic Regression Classifier configuration
# EMPTY
####################################################################
# Standard RBM configuration
NV = 784 			# number of visible units
NH = 250    		# number of hidden units
# Train parameters
EPOCHS = 50         # number of training epochs
ALPHA = [0.01]		# learning rate
M = [0.0]			# momentum parameter
BATCH_SIZE = 50     # size of each batch
GIBBS_K = 1			# number of gibbs sampling steps
ALPHA_UPDATE_RULE = 'constant'  # type of update rule for the learning rate
MOMENTUM_UPDATE_RULE = 'constant'  # type of update rule for the momentum
VERBOSE = True  # if true, a progress bar and a reconstructed sample for each epoch will be shown
OUTFILE = 'models/rbm_layer2.json'  # outfile to save the configuration of the rbm after training
# Mind reader
FANTASY_K = 1000        # number of gibbs sampling steps for reading the mind of the rbm
# Weights image
HS = 20  # how many hidden unit weights to save
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
M_ALPHA = [0.01]    # learning rate
M_M = [0.0]				# momentum parameter
M_BATCH_SIZE = 25     # size of each batch
M_GIBBS_K = 1			# number of gibbs sampling steps
M_ALPHA_UPDATE_RULE = 'constant'  # type of update rule for the learning rate
M_MOMENTUM_UPDATE_RULE = 'constant'  # type of update rule for the momentum
M_VERBOSE = True  # if true, a progress bar and a reconstructed sample for each epoch will be shown
M_OUTFILE = 'models/mrbm.json'  # outfile to save the configuration of the rbm after training
####################################################################
# Gaussian RBM configuration
GAUSS_NV = 784            # number of visible units
GAUSS_NH = 250            # number of hidden units
# Train parameters
G_EPOCHS = 100         # number of training epochs
G_ALPHA = [0.0001]      # learning rate
G_M = [0.0]             # momentum parameter
G_BATCH_SIZE = 50     # size of each batch
G_GIBBS_K = 1         # number of gibbs sampling steps
G_ALPHA_UPDATE_RULE = 'constant'  # type of update rule for the learning rate
G_MOMENTUM_UPDATE_RULE = 'constant'  # type of update rule for the momentum
G_VERBOSE = True  # if true, a progress bar and a reconstructed sample for each epoch will be shown
G_OUTFILE = 'models/grbm.json'  # outfile to save the configuration of the rbm after training
####################################################################
# Deep Belief Network configuration
DBN_LAYERS = [784, 500, 500]
DBN_INPUT_RBMS = ['models/to be in the project/rbm.json', 'models/to be in the project/rbm_layer2.json']
DBN_LAST_LAYER = 1000
# Train parameters
DBN_EPOCHS = 1          # number of training epochs
DBN_ALPHA = [0.05]		# learning rate
DBN_M = [0.0]				# momentum parameter
DBN_BATCH_SIZE = 20     # size of each batch
DBN_GIBBS_K = 1			# number of gibbs sampling steps
DBN_ALPHA_UPDATE_RULE = 'constant'  # type of update rule for the learning rate
DBN_MOMENTUM_UPDATE_RULE = 'constant'  # type of update rule for the momentum
# Supervised fine tuning parameters
DBN_FT_EPOCHS = 50        # number of training epochs
DBN_FT_ALPHA = [0.005]	      # learning rate
DBN_FT_M = [0.05, 0.1]           # momentum
DBN_FT_BATCH_SIZE = 50      # size of each batch
DBN_FT_TOP_GIBBS_K = 1      # number of gibbs sampling steps in the undirected associative memory layer
DBN_FT_ALPHA_UPDATE_RULE = 'constant'  # type of update rule for the learning rate
DBN_FT_MOMENTUM_UPDATE_RULE = 'linear'  # type of update rule for the momentum
# Testing parameters
DBN_TEST_TOP_GIBBS_K = 100

DBN_VERBOSE = True  # if true, a progress bar and a reconstructed sample for each epoch will be shown
# outfile to save the configuration of the rbm after training
DBN_OUTFILES = ['models/rbm1_after_ws.json', 'models/rbm2_after_ws.json', 'models/last_layer_rbm.json']
DBN_PERFORMANCE_OUTFILE = 'models/dbn_performance_metrics.json'  # outfile to save the performance metrics of the rbm
####################################################################
