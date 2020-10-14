import torch
import torch.nn as nn
import torch.nn.functional as F

from .utilities import PAD_idx, UNK_idx

from .ANR_ARL import ANR_ARL
from .ANRS_RatingPred_W import ANRS_RatingPred_W


class WVOE(nn.Module):

	def __init__(self, logger, args, num_users, num_items):	
		super(WVOE, self).__init__()

		self.logger = logger
		self.args = args

		self.num_users = num_users
		self.num_items = num_items

		self.uid_userDoc = nn.Embedding(20, self.args.h1)
		self.uid_userDoc.weight.requires_grad = True

		self.iid_itemDoc = nn.Embedding(20, self.args.h1)
		self.iid_itemDoc.weight.requires_grad = True

		self.wid_wEmbed = nn.Embedding(20, self.args.h1) 
		self.wid_wEmbed.weight.requires_grad = True

		self.ANRS_RatingPred_W = ANRS_RatingPred_W(logger, args)



	def forward(self, batch_uid, batch_iid, batch_label, batch_w, pos_u, pos_v, neg_v, pos_w, db_pos_u, db_pos_v, db_neg_v, db_pos_w, batch_e2_uid, batch_e2_iid, batch_e2_label, verbose = 0):
		batch_userDoc = self.uid_userDoc(batch_uid)
		batch_itemDoc = self.uid_userDoc(batch_iid)

		batch_e2_userDoc = self.uid_userDoc(batch_e2_uid)
		batch_e2_itemDoc = self.uid_userDoc(batch_e2_iid)

		emb_u = self.uid_userDoc(pos_u)
		emb_v = self.uid_userDoc(pos_v)
		emb_neg_v = self.uid_userDoc(neg_v)

		db_emb_u = self.uid_userDoc(db_pos_u)
		db_emb_v = self.uid_userDoc(db_pos_v)
		db_emb_neg_v = self.uid_userDoc(db_neg_v)		
		rating_pred = self.ANRS_RatingPred_W(batch_userDoc, batch_itemDoc, batch_label, batch_w, emb_u, emb_v, emb_neg_v, pos_w, db_emb_u, db_emb_v, db_emb_neg_v, db_pos_w, batch_e2_userDoc, batch_e2_itemDoc, batch_e2_label, verbose = verbose)

		return rating_pred


