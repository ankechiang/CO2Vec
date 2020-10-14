import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm


class ANRS_RatingPred_W(nn.Module):

	def __init__(self, logger, args):

		super(ANRS_RatingPred_W, self).__init__()

		self.logger = logger
		self.args = args

		self.userFC = nn.Linear(self.args.num_aspects * self.args.h1, self.args.h1)
		self.itemFC = nn.Linear(self.args.num_aspects * self.args.h1, self.args.h1)

		self.userFC_Dropout = nn.Dropout(p = self.args.dropout_rate)
		self.itemFC_Dropout = nn.Dropout(p = self.args.dropout_rate)

		self.user_item_rep_dim = self.args.h1

		self.prediction = nn.Linear(2 * self.user_item_rep_dim, 1)
		self.userFC.weight.data.uniform_(0.00, 0.01)
		self.itemFC.weight.data.uniform_(0.00, 0.01)
		self.prediction.weight.data.uniform_(0.00, 0.01)


	def forward(self, userWordEmb, itemWordEmb, batch_label, batch_w, emb_u, emb_v, emb_neg_v, pos_w, emb_db_u, emb_db_v, emb_db_neg_v, db_pos_w, userE2Emb, itemE2Emb, batch_e2_label, verbose = 1):
		### Loss(U) ###
		n = userWordEmb.size(0)
		m = userWordEmb.size(1)
		x = userWordEmb
		y = itemWordEmb

		## o1: ordered link type 1 
		dist = torch.pow(torch.relu(x - y), 2).sum(1)
		rating_pred = torch.squeeze(dist)

		## o2: ordered link type 2
		x_e2 = userE2Emb
		y_e2 = itemE2Emb
		dist_e2 = torch.pow(torch.relu(x_e2 - y_e2), 2).sum(1)
		rating_pred_e2 = torch.squeeze(dist_e2)		

		### cross-link ###
		score = torch.mul( torch.sum(torch.mul(emb_u, emb_v), dim=1), pos_w)   
		score = -F.logsigmoid(-score)
		neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2))
		neg_score = torch.sum( F.sigmoid(neg_score), dim=1)
		
		db_score = torch.mul( torch.sum(torch.mul(emb_db_u, emb_db_v), dim=1), db_pos_w)        
		db_score = -F.logsigmoid(-db_score)
		db_neg_score = torch.bmm(emb_db_neg_v, emb_db_u.unsqueeze(2)) 
		db_neg_score = torch.sum( F.sigmoid(db_neg_score), dim=1)

		return rating_pred, rating_pred_e2, torch.sum(score, dim=0) + torch.mean( neg_score, dim=0), torch.mean(score, dim=0) + torch.mean( neg_score, dim=0) + torch.mean( db_score, dim=0) + torch.sum( db_neg_score, dim=0) #, torch.mean(db_score + db_neg_score)
		
		
