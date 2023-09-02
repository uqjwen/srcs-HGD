import os
import argparse
import numpy as np 
import pickle
class Data_Loader():
	def __init__(self, args):
		self.args = args
		self.doc_vecs = np.load('./data_process/'+self.args.dataset+'_5.npy')
		assert(self.doc_vecs.shape[-1] == self.args.doc_size)
		self.train_pt = 0
		data_path = './data_process/'+self.args.dataset+'.pkl'
		# data_path = './data_process/'+self.args.dataset+'_'+str(args.ratio)+'_'+str(args.num)+'.pkl'
		if not os.path.exists(data_path):

			self.get_user_item()
			self.get_his()
			print('his_len:', self.args.his_len)
			self.get_spec()

			data = {}
			data['pmtt']          = self.pmtt
			data['rating']        = self.rating
			data['user']          = self.user
			data['item']          = self.item
			data['num_user']      = self.num_user
			data['num_item']      = self.num_item
			data['spec_doc']      = self.spec_doc
			data['spec_pos']      = self.spec_pos
			data['spec_msk']      = self.spec_msk
			pickle.dump(data, open(data_path, 'wb'))
			self.spec_doc = self.spec_doc[:,:self.args.his_len]
			self.spec_msk = self.spec_msk[:,:self.args.his_len]
			self.spec_pos = self.spec_pos[:,:self.args.his_len]
		else:
			print('!!! loading data from %s.pkl'%self.args.dataset)
			data = pickle.load(open(data_path, 'rb'))
			self.pmtt          = data['pmtt']         
			self.rating        = data['rating']       
			self.user          = data['user']         
			self.item          = data['item']         
			self.num_user      = data['num_user']     
			self.num_item      = data['num_item']     
			self.spec_doc      = data['spec_doc'][:,:self.args.his_len]
			self.spec_pos      = data['spec_pos'][:,:self.args.his_len]
			self.spec_msk      = data['spec_msk'][:,:self.args.his_len]



		self.data_split()

	def get_user_item(self):
		fr = open('./data_process/'+self.args.dataset+'_5.csv')
		
		user_dic = {}
		item_dic = {}

		user = []
		item = []
		rating = []

		for line in fr.readlines():
			line = line.strip().split('\t')
			user_dic[line[1]] = user_dic.get(line[1], len(user_dic))
			item_dic[line[2]] = item_dic.get(line[2], len(item_dic))
			rating.append(float(line[0]))
			user.append(user_dic[line[1]])
			item.append(item_dic[line[2]])
		self.rating   = np.array(rating)
		self.user     = np.array(user)
		self.item     = np.array(item)
		self.num_user = len(user_dic)
		self.num_item = len(item_dic)
		self.pmtt     = np.random.permutation(len(self.rating))

	def data_split(self):
		self.data_size = len(self.rating)
		self.train_size = int(0.8*self.data_size)
		self.dev_size   = int(0.1*self.data_size)
		self.test_size  = int(0.1*self.data_size)
		self.train_idx  = self.pmtt[:self.train_size]
		self.dev_idx    = self.pmtt[self.train_size:self.train_size+self.dev_size]
		self.test_idx   = self.pmtt[self.train_size+self.dev_size:]


	def get_his(self):
		user_his_dic = {i:[] for i in range(self.num_user)}
		item_his_dic = {i:[] for i in range(self.num_item)}
		for ui, vj in zip(self.user, self.item):
			user_his_dic[ui].append(vj)
			item_his_dic[vj].append(ui)
		self.user_his, self.user_his_mask = self.pad_his(user_his_dic)
		self.item_his, self.item_his_mask = self.pad_his(item_his_dic)

		assert(self.user_his.shape == (self.num_user, self.args.his_len))
		assert(self.item_his.shape == (self.num_item, self.args.his_len))

	def pad_his(self, his):
		ret  = []
		mask = []
		for i in range(len(his)):
			if len(his[i])>self.args.his_len:
				ret.append(his[i][:self.args.his_len])
				mask.append([1]*self.args.his_len)
			else:
				ret.append(his[i]+[0]*(self.args.his_len - len(his[i])))
				mask.append([1]*len(his[i]) + [0]*(self.args.his_len - len(his[i])))
		return np.array(ret), np.array(mask)

	def get_spec(self):
		item_his_dic = {i:[] for i in range(self.num_item)}
		user_his_dic = {i:[] for i in range(self.num_user)}
		for i, (ui, vj, rij) in enumerate(zip(self.user, self.item, self.rating)):
			item_his_dic[vj].append((ui, rij, i))
			user_his_dic[ui].append((vj, rij, i))
		spec_doc = []
		spec_msk = []
		spec_pos = []
		for i,(ui,vj,rij) in enumerate(zip(self.user, self.item, self.rating)):
			tuples = user_his_dic[ui]+item_his_dic[vj]
			doc, pos, msk = self.get_spec_pad(tuples, rij)
			spec_doc.append(doc)
			spec_pos.append(pos)
			spec_msk.append(msk)
			# if not (len(doc)==30 and len(pos)==30 and len(msk)==30):
			# print(doc)
			# print(pos)
			# print(msk)
			# print('\n')
		self.spec_doc = np.array(spec_doc)
		self.spec_pos = np.array(spec_pos)
		self.spec_msk = np.array(spec_msk)

		assert(self.spec_doc.shape == (len(self.rating), 30))
	def get_spec_pad(self, tuples, rating):
		spec_pos = []
		spec_doc = []
		spec_msk = []
		while 1:
			rand_idx = np.random.randint(len(tuples))
			(uv, r, doc_id) = tuples.pop(rand_idx)
			if r==rating:
				spec_pos.append(1)
			else:
				spec_pos.append(0)
			spec_doc.append(doc_id)
			spec_msk.append(1)
			if len(tuples) == 0:
				break

		if len(spec_doc)>30:
			return spec_doc[:30], spec_pos[:30], spec_msk[:30]
		else:
			left = 30 - len(spec_doc)
			return spec_doc+[0]*left,\
			spec_pos+[0]*left,\
			spec_msk+[0]*left



	def reset_train_pt(self):
		self.train_pt = 0
		self.conta_pt = 0
		
	def reset_test_pt(self):
		self.test_pt = 0
	def test_next_batch(self):
		begin = self.args.batch_size*self.test_pt
		end   = min(self.test_size, self.args.batch_size*(self.test_pt+1))
		self.test_pt = self.test_pt+1
		if self.test_pt*self.args.batch_size>=self.test_size:
			self.test_pt = 0

		idx         = self.test_idx[begin:end]
		ui_idx      = self.user[idx]
		vj_idx      = self.item[idx]
		spec_idx    = self.spec_doc[idx]
		spec_msk    = self.spec_msk[idx]
		spec_pos    = self.spec_pos[idx]

		return self.rating[idx],\
		self.user[idx],\
		self.item[idx],\
		self.doc_vecs[idx],\
		self.doc_vecs[spec_idx],\
		spec_msk,\
		spec_pos,\
		spec_idx



	def next_batch(self):
		begin = self.args.batch_size*self.train_pt
		end   = min(self.train_size, self.args.batch_size*(self.train_pt+1))
		self.train_pt = self.train_pt+1
		if self.train_pt*self.args.batch_size>=self.train_size:
			self.train_pt = 0



		idx         = self.train_idx[begin:end]
		ui_idx      = self.user[idx]
		vj_idx      = self.item[idx]
		spec_idx    = self.spec_doc[idx]
		spec_msk    = self.spec_msk[idx]
		spec_pos    = self.spec_pos[idx]

		return self.rating[idx],\
		self.user[idx],\
		self.item[idx],\
		self.doc_vecs[idx],\
		self.doc_vecs[spec_idx],\
		spec_msk,\
		spec_pos
