# Datasets sizes
TRAIN_SET_SIZE = 500
TEST_SET_SIZE = 4000


# Standard RBM configuration
NV = 784 			# number of visible units
NH = 250    		# number of hidden units
# Train parameters
MAX_EPOCHS = 25   # number of training epochs
ALPHA = 0.12		# learning rate
M = 0.0				# momentum parameter
BATCH_SIZE = 20     # size of each batch
GIBBS_K = 1			# number of gibbs sampling steps
VERBOSE = True  # if true, a progress bar and a reconstructed sample for each epoch will be shown
OUTFILE = 'models/rbm.json'  # outfile to save the configuration of the rbm after training

# Gaussian RBM configuration
GAUSS_NV = 784            # number of visible units
GAUSS_NH = 450            # number of hidden units
# Train parameters
G_MAX_EPOCHS = 2000   # number of training epochs
G_ALPHA = 0.001       # learning rate
G_M = 0.0             # momentum parameter
G_BATCH_SIZE = 20     # size of each batch
G_GIBBS_K = 1         # number of gibbs sampling steps
G_VERBOSE = True  # if true, a progress bar and a reconstructed sample for each epoch will be shown
G_OUTFILE = 'models/grbm.json'  # outfile to save the configuration of the rbm after training
