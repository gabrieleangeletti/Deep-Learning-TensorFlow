# Datasets sizes
TRAIN_SET_SIZE = 1000
TEST_SET_SIZE = 10000

####################################################################

# Binary RBM configuration
NV = 784 			# number of visible units
NH = 250    		# number of hidden units
# Train parameters
EPOCHS = 5         # number of training epochs
ALPHA = [0.01]		# learning rate
M = [0.0]			# momentum parameter
BATCH_SIZE = 50     # size of each batch
GIBBS_K = 1			# number of gibbs sampling steps
ALPHA_UPDATE_RULE = 'constant'  # type of update rule for the learning rate
MOMENTUM_UPDATE_RULE = 'constant'  # type of update rule for the momentum
VERBOSE = True  # if true, a progress bar and a reconstructed sample for each epoch will be shown
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
####################################################################

# Deep Belief Network configuration
DBN_LAYERS = [784, 500, 500]
DBN_INPUT_RBMS = ['models/to be in the project/rbm.json', 'models/to be in the project/rbm_layer2.json']
DBN_LAST_LAYER = 400
# Train parameters
DBN_EPOCHS = 1          # number of training epochs
DBN_ALPHA = [0.05]		# learning rate
DBN_M = [0.0]				# momentum parameter
DBN_BATCH_SIZE = 20     # size of each batch
DBN_GIBBS_K = 1			# number of gibbs sampling steps
DBN_ALPHA_UPDATE_RULE = 'constant'  # type of update rule for the learning rate
DBN_MOMENTUM_UPDATE_RULE = 'constant'  # type of update rule for the momentum
# Supervised fine tuning wake-sleep parameters
DBN_FT_EPOCHS = 100        # number of training epochs
DBN_FT_ALPHA = [0.0005]	      # learning rate
DBN_FT_M = [0.05, 0.1]           # momentum
DBN_FT_BATCH_SIZE = 100      # size of each batch
DBN_FT_TOP_GIBBS_K = 1      # number of gibbs sampling steps in the undirected associative memory layer
DBN_FT_ALPHA_UPDATE_RULE = 'constant'  # type of update rule for the learning rate
DBN_FT_MOMENTUM_UPDATE_RULE = 'constant'  # type of update rule for the momentum
# Testing parameters wake-sleep
DBN_TEST_TOP_GIBBS_K = 300
# Supervised fine tuning backpropagation parameters
DBN_BP_SOFTMAX_LAYER = 10
DBN_BP_EPOCHS = 50        # number of training epochs
DBN_BP_ALPHA = [0.001]        # learning rate
DBN_BP_M = [0.05, 0.1]         # momentum
DBN_BP_BATCH_SIZE = 50      # size of each batch
DBN_BP_ALPHA_UPDATE_RULE = 'constant'  # type of update rule for the learning rate
DBN_BP_MOMENTUM_UPDATE_RULE = 'constant'  # type of update rule for the momentum

DBN_VERBOSE = True  # if true, a progress bar and a reconstructed sample for each epoch will be shown
# outfile to save the configuration of the rbm after training
DBN_OUTFILES = ['models/rbm1_after_ws.json', 'models/rbm2_after_ws.json', 'models/last_layer_rbm.json']
DBN_BP_OUTFILES = ['models/rbm1_after_bp.json', 'models/rbm2_after_bp.json']
# outfile to save the performance metrics of the rbm
DBN_WS_PERFORMANCE_OUTFILE = 'models/dbn_performance_wakesleep.json'
# outfile to save the performance metrics of the rbm
DBN_BP_PERFORMANCE_OUTFILE = 'models/dbn_performance_backprop.json'
####################################################################
