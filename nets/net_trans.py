import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math
from ipdb import set_trace

from torch import Tensor
from typing import Optional, Any
torch.set_num_threads(2)
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

def _get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):

	def __init__(self, encoder_layer, num_layers, norm=None):
		super(Encoder, self).__init__()
		self.layers = _get_clones(encoder_layer, num_layers)
		self.num_layers = num_layers
		self.norm1 = nn.LayerNorm(512)
		self.norm2 = nn.LayerNorm(512)
		self.norm = norm

	def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None):
		output_a = src_a
		output_v = src_v

		for i in range(self.num_layers):
			output_a = self.layers[i](src_a, src_v, src_mask=mask,
									src_key_padding_mask=src_key_padding_mask)
			output_v = self.layers[i](src_v, src_a, src_mask=mask,
									  src_key_padding_mask=src_key_padding_mask)

		if self.norm:
			output_a = self.norm1(output_a)
			output_v = self.norm2(output_v)

		return output_a, output_v

class HANLayer(nn.Module):

	def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
		super(HANLayer, self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
		self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

		# Implementation of Feedforward model
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout11 = nn.Dropout(dropout)
		self.dropout12 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

		self.activation = nn.ReLU()

	def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
		"""Pass the input through the encoder layer.

		Args:
			src: the sequnce to the encoder layer (required).
			src_mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		src_q = src_q.permute(1, 0, 2)
		src_v = src_v.permute(1, 0, 2)
		src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,
							  key_padding_mask=src_key_padding_mask)[0]
		src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
							  key_padding_mask=src_key_padding_mask)[0]
		src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
		src_q = self.norm1(src_q)

		src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
		src_q = src_q + self.dropout2(src2)
		src_q = self.norm2(src_q)
		return src_q.permute(1, 0, 2)


class MMIL_Net(nn.Module):

	def __init__(self, opt):
		super(MMIL_Net, self).__init__()

		self.fc_prob = nn.Linear(512, 25)
		self.fc_frame_att = nn.Linear(512, 25)
		self.fc_av_att = nn.Linear(512, 25)
		
		self.fc_a =  nn.Linear(128, 512)

		self.fc_v = nn.Linear(2048, 512)
		self.fc_st = nn.Linear(512, 512)
		self.fc_fusion = nn.Linear(1024, 512)


		self.fc_occ_class_a = nn.Linear(512, 25*opt.occ_dim)
		self.fc_occ_class_v = nn.Linear(512, 25*opt.occ_dim)
		self.fc_occ_class_av = nn.Linear(512*2, 25*opt.occ_dim)

		self.fc_occ_v_q = nn.Linear(opt.occ_dim, opt.occ_dim)
		self.fc_occ_v_k = nn.Linear(opt.occ_dim, opt.occ_dim)
		self.fc_occ_v_v = nn.Linear(opt.occ_dim, opt.occ_dim)

		self.fc_occ_a_q = nn.Linear(opt.occ_dim, opt.occ_dim)
		self.fc_occ_a_k = nn.Linear(opt.occ_dim, opt.occ_dim)
		self.fc_occ_a_v = nn.Linear(opt.occ_dim, opt.occ_dim)


		self.fc_occ_av_q = nn.Linear(opt.occ_dim, opt.occ_dim)
		self.fc_occ_av_k = nn.Linear(opt.occ_dim, opt.occ_dim)
		self.fc_occ_av_v = nn.Linear(opt.occ_dim, opt.occ_dim)

		self.fc_occ_prob_a = nn.Linear(opt.occ_dim, 1)
		self.fc_occ_prob_v = nn.Linear(opt.occ_dim, 1)

		self.fc_occ_frame_prob = nn.Linear(opt.occ_dim, 1)
		self.fc_occ_modality_prob = nn.Linear(opt.occ_dim, 1)


		self.hat_encoder = Encoder(HANLayer(d_model=512, nhead=1, dim_feedforward=opt.forward_dim, dropout=opt.prob_drop), num_layers=1)


		self.av_class_encoder = Encoder(HANLayer(d_model=opt.occ_dim, nhead=1, dim_feedforward=opt.occ_dim, dropout=opt.prob_drop_occ), num_layers=1)

		self.frame_att_encoder = Encoder(HANLayer(d_model=opt.occ_dim, nhead=1, dim_feedforward=opt.occ_dim, dropout=opt.prob_drop_occ), num_layers=1)
		self.modality_encoder = Encoder(HANLayer(d_model=opt.occ_dim, nhead=1, dim_feedforward=opt.occ_dim, dropout=opt.prob_drop_occ), num_layers=1)


	def co_occurrence(self, input, opt):

		
		x1_class, x2_class= input



		q_a = self.fc_occ_a_q(x1_class)
		k_a = self.fc_occ_a_k(x1_class)
		v_a = self.fc_occ_a_v(x1_class)


		q_v = self.fc_occ_v_q(x2_class)
		k_v = self.fc_occ_v_k(x2_class)
		v_v = self.fc_occ_v_v(x2_class)

		q_av = torch.cat((q_a, q_v),dim=1)
		k_av = torch.cat((k_a, k_v),dim=1)
		v_av = torch.cat((v_a, v_v),dim=1)

		att_1 = F.softmax(torch.bmm(q_av,k_av.permute(0,2,1))/np.sqrt(opt.occ_dim), dim=-1)
		res_av = torch.bmm(att_1,v_av).view(x1_class.size(0),50,-1)



		return res_av

	def forward(self, audio, visual, visual_st, opt, a_refine=None,v_refine=None, rand_idx=None, sample_idx=None, label=None, target=None):

		x1_audio = self.fc_a(audio)

		# 2d and 3d visual feature fusion
		vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
		vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
		vid_st = self.fc_st(visual_st)
		x2 = torch.cat((vid_s, vid_st), dim =-1)
		x2_visual = self.fc_fusion(x2)

		
		x1,x2 = self.hat_encoder(x1_audio, x2_visual)

		

		

		x1_class = self.fc_occ_class_a(x1).relu().view(x1.size(0)*10,25,opt.occ_dim)
		x2_class = self.fc_occ_class_v(x2).relu().view(x2.size(0)*10,25,opt.occ_dim)

		
		x1_class_att, x2_class_att = self.av_class_encoder(x1_class, x2_class)


		x1_class_att = self.fc_occ_prob_a(x1_class_att).sigmoid().view(x1.size(0),10,1,-1)
		x2_class_att = self.fc_occ_prob_v(x2_class_att).sigmoid().view(x2.size(0),10,1,-1)
		frame_prob = torch.cat((x1_class_att, x2_class_att), dim=-2)




		x = torch.cat([x1.unsqueeze(-2), x2.unsqueeze(-2)], dim=-2)



		frame_att = torch.softmax(self.fc_frame_att(x), dim=1)

		av_att = torch.softmax(self.fc_av_att(x), dim=2)

		temporal_prob = (frame_att * frame_prob)





		global_prob = (temporal_prob*av_att).sum(dim=2).sum(dim=1)




		a_prob = temporal_prob[:, :, 0, :].sum(dim=1)
		v_prob =temporal_prob[:, :, 1, :].sum(dim=1)


		## my sim ##
		if rand_idx is not None:
			gg_weight = self.fc_frame_att(x)
			
			x1_target = a_refine.float()
			x2_target = v_refine.float()

			a_related_w = gg_weight[:,:,0,:] * x1_target.unsqueeze(1)
			a_related_w = a_related_w.sum(-1).unsqueeze(-1)
			a_related_w = torch.softmax(a_related_w, dim=1)

			v_related_w = gg_weight[:,:,1,:] * x2_target.unsqueeze(1)
			v_related_w = v_related_w.sum(-1).unsqueeze(-1)
			v_related_w = torch.softmax(v_related_w, dim=1)

			

			if opt.is_a_ori:
				a_agg = torch.bmm(x1_audio.permute(0,2,1), a_related_w)
			else:
				a_agg = torch.bmm(x1.permute(0,2,1), a_related_w)

			if opt.is_v_ori:
				v_agg = torch.bmm(x2_visual.permute(0,2,1), v_related_w)
			else:
				v_agg = torch.bmm(x2.permute(0,2,1), v_related_w)

			

			xx1 = F.normalize(a_agg, p=2, dim=1)        
			xx2 = F.normalize(v_agg, p=2, dim=1)

			sims= torch.mm(xx1.squeeze(-1), xx2.squeeze(-1).permute(1,0)) / opt.tmp

			return global_prob, a_prob, v_prob, frame_prob, sims


		# dumming output: x1,x2
		return global_prob, a_prob, v_prob, frame_prob, (x1, x2) 


class CMTLayer(nn.Module):

	def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
		super(CMTLayer, self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
		# Implementation of Feedforward model
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

		self.activation = nn.ReLU()

	def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
		r"""Pass the input through the encoder layer.

		Args:
			src: the sequnce to the encoder layer (required).
			src_mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		src2 = self.self_attn(src_q, src_v, src_v, attn_mask=src_mask,
							  key_padding_mask=src_key_padding_mask)[0]
		src_q = src_q + self.dropout1(src2)
		src_q = self.norm1(src_q)

		src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
		src_q = src_q + self.dropout2(src2)
		src_q = self.norm2(src_q)
		return src_q
