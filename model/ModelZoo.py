import torch
import torch.nn as nn

from .utilities import *
from .WVOE import WVOE
from .VOE import VOE

import numpy as np


class ModelZoo():

	def __init__(self, logger, args, timer):

		self.mdl = None
		self.logger = logger
		self.args = args
		self.timer = timer
		
		self.num_users, self.num_items = 5, 5


	def createAndInitModel(self):
		self.timer.startTimer("init")
		self.logger.log("\nCreating model (Selected Model: {})..".format( self.args.model ))

		self.createModel()

		# Port to GPU, if necessary
		if(self.args.use_cuda):
			self.mdl.cuda()
			self.logger.log("[args.use_cuda: {}] Model is on the GPU! (args.gpu: {}, torch.cuda.current_device(): {})".format(
				self.args.use_cuda, self.args.gpu, torch.cuda.current_device() ))

		self.logger.log("Model created! {}".format( self.timer.getElapsedTimeStr("init", conv2Mins = True) ))

		'''
		Model Initialization
		'''
		self.initModel()

		# Cleanup
		if(self.args.use_cuda):
			torch.cuda.empty_cache()

		self.logger.log("\nInitialization Complete.. {}".format( self.timer.getElapsedTimeStr("init", conv2Mins = True) ))
		return self.mdl


	def createModel(self):
		self.args.vocab_size = self.args.vocab_size 
		if(self.args.model == "VOE"):
			self.mdl = VOE(self.logger, self.args, self.num_users, self.num_items)
		elif (self.args.model == "WVOE"):
			self.mdl = WVOE(self.logger, self.args, self.num_users, self.num_items)

	def initModel(self):
		# if(self.args.model == "VOE"):
		self.initVOE()


	def initVOE(self):
		self.loadDocs()
		self.loadWordEmbeddings()


	def loadDocs(self):
		uid_userDoc_path = "{}{}{}".format( self.args.input_dir, self.args.dataset, fp_uid_userDoc )
		iid_itemDoc_path = "{}{}{}".format( self.args.input_dir, self.args.dataset, fp_iid_itemDoc )

		# User Documents
		np_uid_userDoc = np.load( uid_userDoc_path )

		if(self.args.use_cuda):
			np_uid_userDoc = torch.cuda.FloatTensor(len(np_uid_userDoc), self.args.h1).uniform_(0.00,0.01) #add
		else:
			if self.args.groundtruth in ["_random", ""]:
				np_uid_userDoc = torch.Tensor(len(np_uid_userDoc), self.args.h1).uniform_(0.00,0.01) #add
			elif self.args.groundtruth  == "_pre": np_uid_userDoc = torch.Tensor(np_uid_userDoc)
		self.mdl.uid_userDoc.weight.data = np_uid_userDoc
		# Item Documents
		np_iid_itemDoc = np.load( iid_itemDoc_path )
		if(self.args.use_cuda):
			np_iid_itemDoc = torch.cuda.FloatTensor(len(np_iid_itemDoc), self.args.h1).uniform_(0.00,0.01) #add
		else:
			if self.args.groundtruth in ["_random", ""]:
				np_iid_itemDoc = torch.Tensor(len(np_iid_itemDoc), self.args.h1).uniform_(0.00,0.01) #add
			elif self.args.groundtruth  == "_pre": np_iid_itemDoc = torch.Tensor(np_iid_itemDoc)
		print(self.mdl.iid_itemDoc.weight.data.shape)
		self.mdl.iid_itemDoc.weight.data = np_uid_userDoc
		print(self.mdl.iid_itemDoc.weight.data.shape)
		del np_uid_userDoc
		del np_iid_itemDoc


	# Load/Randomly initialize the word embeddings
	def loadWordEmbeddings(self):

		if(self.args.pretrained_src == 1 or self.args.pretrained_src == 2):
			if(self.args.pretrained_src == 1):
				wid_wEmbed_path = "{}{}{}".format( self.args.input_dir, self.args.dataset, fp_wid_wordEmbed )
			elif(self.args.pretrained_src == 2):
				wid_wEmbed_path = "{}{}{}".format( self.args.input_dir, self.args.dataset, fp_wid_wordEmbed )

			self.logger.log("\nLoading pretrained word embeddings from \"{}\"..".format( wid_wEmbed_path ))
			np_wid_wEmbed = np.load( wid_wEmbed_path )

			self.mdl.wid_wEmbed.weight.data.copy_(torch.from_numpy(np_wid_wEmbed))
			self.logger.log("Pretrained word embeddings loaded! [wid_wEmbed: {}]".format( np_wid_wEmbed.shape ))
			del np_wid_wEmbed

		elif(self.args.pretrained_src == 3):
			wid_1hot_path = "{}{}{}".format( self.args.input_dir, self.args.dataset, fp_wid_wordEmbed )
			self.logger.log("\nLoading one-hot word vectors from \"{}\"..".format( wid_1hot_path ))
			np_wid_1hot = np.load(wid_1hot_path)

			self.mdl.wid_wEmbed.weight.data.copy_(torch.from_numpy(np_wid_1hot))
			self.logger.log("Pretrained word embeddings loaded! [wid_wEmbed: {}]".format( np_wid_1hot.shape ))
			del np_wid_1hot

		else:
			rand_uniform_dist_min = 0.00
			rand_uniform_dist_max = 0.01
			self.mdl.wid_wEmbed.weight.data.uniform_(rand_uniform_dist_min, rand_uniform_dist_max)
			self.logger.log("\nWord embeddings are randomly initialized using random uniform distribution from [{:.2f}, {:.2f}]..".format(
				rand_uniform_dist_min, rand_uniform_dist_max ))
		self.mdl.wid_wEmbed.weight.data[PAD_idx].fill_(0)
		self.mdl.wid_wEmbed.weight.data[UNK_idx].fill_(0)


	'''
	Optimizer & Loss Function
	'''

	# Optimizer
	def selectOptimizer(self, optimizer = "Adam", learning_rate = 2E-3, L2_reg = 0):
		opt_params = self.VOE_Params()
		self.optimizer = "Adam"
		return optim.Adam(opt_params, lr = learning_rate)



	# Loss Function
	def selectLossFunction(self, loss_function = "HingeEmbeddingLoss"):
		self.loss_function = "HingeEmbeddingLoss"
		return nn.HingeEmbeddingLoss(margin=2) 


	# Apply L2 regularization to the User Bias & Item Bias
	def VOE_Params(self):

		normalParams, paramsWithDiffLR, paramsWithL2Reg = [], [], []
		lstDiffLRParamNames = []
		lstL2RegParamNames = []

		for name, param in self.mdl.named_parameters():

			if(not param.requires_grad):
				continue	
			normalParams.append(param)

		if(lstDiffLRParamNames):
			self.logger.log("\nParameters that are fine-tuned using a smaller LR (LR: {}):\n{}".format(
				(self.args.learning_rate * self.args.ARL_lr), ", ".join(lstDiffLRParamNames) ))

		if(lstL2RegParamNames):
			self.logger.log("\nParameters with L2 Regularization (Regularization Strength: {}):\n{}".format(
				self.args.L2_reg, ", ".join(lstL2RegParamNames) ))

		return [{'params': paramsWithL2Reg, 'lr': self.args.learning_rate, 'weight_decay': self.args.L2_reg},
				{'params': paramsWithDiffLR, 'lr': (self.args.learning_rate * self.args.ARL_lr)},
				{'params': normalParams, 'lr': self.args.learning_rate}]


