import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utilities import *
from model.ModelZoo import ModelZoo
from model.Logger import Logger
from model.Timer import Timer

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from tqdm import tqdm
import argparse
from datetime import datetime
import random


parser = argparse.ArgumentParser()

# Dataset & Model
parser.add_argument("-d", 		dest = "dataset", 	type = str, default = "musical_instruments", 	help = "Dataset for Running Experiments (Default: musical_instruments)")
parser.add_argument("-folder", 		dest = "folder", 	type = str, default = "within_company_transition_w_position", 	help = "Dataset for Running Experiments (Default: within_company_transition_w_position)")
parser.add_argument("-m", 		dest = "model", 	type = str, default = "WVOE", 					help = "Model Name, Default: CO2V (i.e., WVOE)")

# General Hyperparameters
parser.add_argument("-reg", 			dest = "regularizer", 	type = float, 	default = 0.0,	 		help = "Regularizer Weight (Default: 0.0)")
parser.add_argument("-ws", 			dest = "window_size", 	type = int, 	default = 5,	 		help = "Window Size (Default: 5)")
parser.add_argument("-bs", 			dest = "batch_size", 	type = int, 	default = 256,	 		help = "Batch Size (Default: 128)")
parser.add_argument("-e", 			dest = "epochs", 		type = int, 	default = 25, 			help = "Number of Training Epochs (Default: 25)")
parser.add_argument("-lr", 			dest = "learning_rate", type = float, 	default = 1E-3, 		help = "Learning Rate (Default: 0.002, i.e 2E-3)")
parser.add_argument("-opt", 		dest = "optimizer", 	type = str, 	default = "Adam", 		help = "Optimizer, e.g. Adam|RMSProp|SGD (Default: Adam)")
parser.add_argument("-loss_func", 	dest = "loss_function", type = str, 	default = "MarginRankingLoss", 	help = "Loss Function, e.g. MSELoss|L1Loss (Default: MSELoss)")
parser.add_argument("-dr",			dest = "dropout_rate", 	type = float, 	default = 0.0, 			help = "Dropout rate (Default: 0.5)")
parser.add_argument("-alpha",		dest = "margin", 		type = float, 	default = 1.0, 			help = "Margin (Default: 1.0)")

# Dataset-Specific Settings (Document Length, Vocabulary Size, Dimensionality of the Embedding Layer, Source of Pretrained Word Embeddings)
parser.add_argument("-MDL", 		dest = "max_doc_len", 		type = int, 	default = 5, 		help = "Maximum User/Item Document Length (Default: 500)")
parser.add_argument("-v", 			dest = "vocab_size", 		type = int, 	default = 5, 	help = "Vocabulary Size (Default: 50000)")
parser.add_argument("-WED", 		dest = "word_embed_dim", 	type = int, 	default = 5, 		help = "Number of Dimensions for the Word Embeddings (Default: 300)")
parser.add_argument("-p", 			dest = "pretrained_src", 	type = int, 	default = 4,		help = "Source of Pretrained Word Embeddings? \
	0: Randomly Initialized (Random Uniform Dist. from [-0.01, 0.01]), 1: w2v (Google News, 300d), 2: GloVe (6B, 400K, 100d) (Default: 1) , 3: onehot")
parser.add_argument("-gt", 			dest = "groundtruth", 	type = str, 	default = "",	help = "Source of Groundtruth")

# VOE Hyperparameters
parser.add_argument("-K", 		dest = "num_aspects", 	type = int, 	default = 1, 	help = "Number of Aspects (Default: 5)")
parser.add_argument("-h1", 		dest = "h1", 			type = int, 	default = 10, 	help = "Dimensionality of the Aspect-level Representations (Default: 10)")
parser.add_argument("-c", 		dest = "ctx_win_size", 	type = int, 	default = 1, 	help = "Window Size (i.e. Number of Words) for Calculating Attention (Default: 3)")
parser.add_argument("-L2_reg", 	dest = "L2_reg", 		type = float, 	default = 1E-6, help = "L2 Regularization for User & Item Bias (Default: 1E-6)")


# Miscellaneous
parser.add_argument("-rs", 	dest = "random_seed", 			type = int, default = 1337, help = "Random Seed (Default: 1337)")
parser.add_argument("-dc", 	dest = "disable_cuda", 			type = int, default = 0, 	help = "Disable CUDA? (Default: 0, i.e. run using GPU (if available))")
parser.add_argument("-gpu", dest = "gpu", 					type = int, default = 6, 	help = "Which GPU to use? (Default: 0)")
parser.add_argument("-vb", 	dest = "verbose", 				type = int, default = 0, 	help = "Show debugging/miscellaneous information? (Default: 0, i.e. Disabled)")
parser.add_argument("-die", dest = "disable_initial_eval", 	type = int, default = 0, 	help = "Disable initial Dev/Test evaluation? (Default: 0, i.e. Disabled)")
parser.add_argument("-sm", 	dest = "save_model", 			type = str, default = "", 	help = "Specify the file name for saving model! (Default: "", i.e. Disabled)")

args = parser.parse_args()


# Check for availability of CUDA and execute on GPU if possible
args.use_cuda = True
args.use_cuda = not args.disable_cuda and torch.cuda.is_available()
del args.disable_cuda


# Initial Setup
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

if(args.use_cuda):
	select_gpu(args.gpu)
	torch.cuda.set_device(args.gpu)
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.deterministic = True
	torch.cuda.manual_seed(args.random_seed)
	print("\n[args.use_cuda: {}] The program will be executed on the GPU!!".format( args.use_cuda ))
else:
	print("\n[args.use_cuda: {}] The program will be executed on the CPU!!".format( args.use_cuda ))


# Timer & Logging
timer = Timer()
timer.startTimer()

uuid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
args.input_dir = "./datasets/{}/".format( args.dataset )
args.out_dir = "./experimental_results/{} - {}/".format( args.dataset, args.model )
log_path = "{}{}-{}".format(args.out_dir, uuid, 'logs.txt')
logger = Logger(args.out_dir, log_path, args)


# Optional: Saving Model
if(args.save_model != ""):
	saved_models_dir = "./__saved_models__/{} - {}/".format( args.dataset, args.model )
	mkdir_p(saved_models_dir)
	model_path = "{}{}_{}.pth".format( saved_models_dir, args.save_model.strip(), args.random_seed )


# Create model
mdlZoo = ModelZoo(logger, args, timer)
mdl = mdlZoo.createAndInitModel()
# Loss Function, Custom Regularizers, Optimizer
criterion = mdlZoo.selectLossFunction(loss_function = args.loss_function)
opt = mdlZoo.selectOptimizer(optimizer = args.optimizer, learning_rate = args.learning_rate, L2_reg = args.L2_reg)
logger.log("\nOptimizer: {}, Loss Function: {}".format( mdlZoo.optimizer, mdlZoo.loss_function ))

# Load training/validation/testing sets
train_set, train_loader, dev_set, dev_loader, test_set, test_loader = loadTrainDevTest(logger, args)
train_pos_set, train_pos_loader, dev_pos_set, dev_pos_loader, test_pos_set, test_pos_loader = loadCrossTrainDevTest(logger, args)
train_double_pos_set, train_double_pos_loader, dev_double_pos_set, dev_double_pos_loader, test_double_pos_set, test_double_pos_loader = loadDoubleCrossTrainDevTest(logger, args)
train_e2_set, train_e2_loader, dev_e2_set, dev_e2_loader, test_e2_set, test_e2_loader = loadOtherTrainDevTest(logger, args)
logger.log("Train/Dev/Test splits loaded! {}".format( timer.getElapsedTimeStr("init", conv2Mins = True) ))



# For evaluation
def evaluate(mdl, pos_loader, db_pos_loader, set_loader, set_e2_loader, epoch_num = -1, use_cuda = True, phase = "Dev", print_txt = True):
	all_rating_true = []
	all_rating_pred = []
	errs = []
	for batch_num, ((batch_uid, batch_iid, batch_rating), (pos_u, pos_v, neg_v, pos_w), (db_pos_u, db_pos_v, db_neg_v, db_pos_w), (batch_e2_uid, batch_e2_iid, batch_e2_rating)) in enumerate(zip(set_loader, pos_loader, db_pos_loader, set_e2_loader)):
		mdl.eval()

		batch_uid = to_var(batch_uid, use_cuda = use_cuda, phase = phase)
		batch_iid = to_var(batch_iid, use_cuda = use_cuda, phase = phase)
		batch_rating = to_var(batch_rating, use_cuda = use_cuda, phase = phase)

		batch_e2_uid = to_var(batch_e2_uid, use_cuda = use_cuda, phase = phase)
		batch_e2_iid = to_var(batch_e2_iid, use_cuda = use_cuda, phase = phase)
		batch_e2_rating = to_var(batch_e2_rating, use_cuda = use_cuda, phase = phase)

		pos_u = to_var(pos_u, use_cuda = use_cuda, phase = phase)
		pos_v = to_var(pos_v, use_cuda = use_cuda, phase = phase)
		neg_v = to_var(neg_v, use_cuda = use_cuda, phase = phase)
		pos_w = to_var(pos_w, use_cuda = use_cuda, phase = phase)


		db_pos_u = to_var(db_pos_u, use_cuda = use_cuda, phase = phase)
		db_pos_v = to_var(db_pos_v, use_cuda = use_cuda, phase = phase)
		db_neg_v = to_var(db_neg_v, use_cuda = use_cuda, phase = phase)
		db_pos_w = to_var(db_pos_w, use_cuda = use_cuda, phase = phase)

		(rating_pred, rating_pred_e2, network_proximity_c, network_proximity_s) = mdl(batch_uid, batch_iid, batch_rating, pos_u, pos_v, neg_v, pos_w, db_pos_u, db_pos_v, db_neg_v, db_pos_w, batch_e2_uid, batch_e2_iid, batch_e2_rating) 
		violation_e1 = criterion(rating_pred.float(), batch_rating.float())
		violation_e2 = criterion(rating_pred_e2.float(), batch_e2_rating.float())
		err = torch.add( torch.add( torch.add( violation_e1, network_proximity_c), network_proximity_s), violation_e2)
		errs.append(err.data.item())
	return np.mean(errs)
	



# Model Information
generate_mdl_summary(mdl, logger)


lstTrainingLoss, lstLossOc, lstLossOs, lstLossLcs, lstLossLsc, lstRegs= [], [], [], [], [], []
lstDevMSE = []
lstTestMSE = []

timer.startTimer("training")
current_learning_rate = args.learning_rate
warm_up_steps = args.epochs/6
for epoch_num in range(args.epochs):
	losses, Oc, Os, Lcs, Lsc, Rs = [], [], [], [], [], []
	for batch_num, ((batch_uid, batch_iid, batch_rating), (pos_u, pos_v, neg_v, pos_w), (db_pos_u, db_pos_v, db_neg_v, db_pos_w), (batch_uid_e2, batch_iid_e2, batch_rating_e2)) in enumerate(zip(train_loader, train_pos_loader, train_double_pos_loader, train_e2_loader)):

		# Set to training mode, zero out the gradients
		mdl.train()
		opt.zero_grad()

		batch_uid = to_var(batch_uid, use_cuda = args.use_cuda)
		batch_iid = to_var(batch_iid, use_cuda = args.use_cuda)
		rating_true = to_var(batch_rating, use_cuda = args.use_cuda)

		batch_uid_e2 = to_var(batch_uid_e2, use_cuda = args.use_cuda)
		batch_iid_e2 = to_var(batch_iid_e2, use_cuda = args.use_cuda)
		rating_true_e2 = to_var(batch_rating_e2, use_cuda = args.use_cuda)	

		pos_u = to_var(pos_u, use_cuda = args.use_cuda)
		pos_v = to_var(pos_v, use_cuda = args.use_cuda)
		neg_v = to_var(neg_v, use_cuda = args.use_cuda)
		pos_w = to_var(pos_w, use_cuda = args.use_cuda)

		db_pos_u = to_var(db_pos_u, use_cuda = args.use_cuda)
		db_pos_v = to_var(db_pos_v, use_cuda = args.use_cuda)
		db_neg_v = to_var(db_neg_v, use_cuda = args.use_cuda)
		db_pos_w = to_var(db_pos_w, use_cuda = args.use_cuda)

		(rating_pred, rating_pred_e2, network_proximity_c, network_proximity_s) = mdl(batch_uid, batch_iid, rating_true, pos_u, pos_v, neg_v, pos_w, db_pos_u, db_pos_v, db_neg_v, db_pos_w, batch_uid_e2, batch_iid_e2, rating_true_e2)
		
		violation_e1 = criterion(rating_pred.float(), rating_true.float())#.mean()
		violation_e2 = criterion(rating_pred_e2.float(), rating_true_e2.float())#.mean()

		loss = (violation_e1 + violation_e2 ) / 2 #violation_e2 + 
		loss.backward()
		opt.step()


		losses.append(loss.data.item())
		Oc.append(violation_e1.data.item())
		Os.append(violation_e2.data.item())
		Lcs.append(network_proximity_c.data.item())
		Lsc.append(network_proximity_s.data.item())

	trainingLoss = np.mean(losses)
	lstTrainingLoss.append( trainingLoss )


	### L1 L2 L3
	trainOc = np.mean(Oc)
	trainOs = np.mean(Os)
	trainLcs = np.mean(Lcs)
	trainLsc = np.mean(Lsc)

	lstLossOc.append(trainOc)
	lstLossOs.append(trainOs)
	lstLossLcs.append(trainLcs)
	lstLossLsc.append(trainLsc)
	
	if epoch_num >= warm_up_steps:
		current_learning_rate = current_learning_rate / 10
		print('Change learning_rate to %f at epoch_num %d' % (current_learning_rate, epoch_num))
		warm_up_steps = warm_up_steps * 3

	logger.log("\n[Epoch {:d}/{:d}] Training Loss: {:.5f}\t{}".format( epoch_num + 1, args.epochs, trainingLoss, timer.getElapsedTimeStr("training", conv2HrsMins = True) ))


np.save("./weights/"+args.dataset+"lstLossOc."+str(args.h1)+args.groundtruth+".npy", lstLossOc)
np.save("./weights/"+args.dataset+"lstLossOs."+str(args.h1)+args.groundtruth+".npy", lstLossOs)
np.save("./weights/"+args.dataset+"lstLossLcs."+str(args.h1)+args.groundtruth+".npy", lstLossLcs)
np.save("./weights/"+args.dataset+"lstLossLsc."+str(args.h1)+args.groundtruth+".npy", lstLossLsc)
np.save("./weights/"+args.dataset+"lstTrainingLoss."+str(args.h1)+args.groundtruth+".npy", lstTrainingLoss)
np.save("./weights/"+args.dataset+"uid_userDoc.weight.data."+str(args.h1)+args.groundtruth+".npy", mdl.uid_userDoc.weight.data.cpu() )
