import tensorflow as tf 
import pickle 
import argparse
import numpy as np 
import sys
import os
from Data_Loader import Data_Loader
from Model import Model

def debug(model, data_loader, args):
	path = './ckpt/'+args.dataset+'/'+"tradeoff_"+str(args.tradeoff)+'_scalar_'+str(args.scalar)+'_hislen_'+str(args.his_len)+'/'
	if not os.path.exists(path):
		print('model not exists!!!')
		return 
	ckpt = tf.train.Checkpoint(Model = model)
	ckpt.restore(tf.train.latest_checkpoint(path))
	total_batch = int(data_loader.train_size/args.batch_size)+1
	# data_loader.reset_train_pt()
	# for b in range(total_batch):
	# 	r, u, i, d, spec_doc, spec_msk, spec_pos = data_loader.next_batch()
	# 	model.get_rate_loss(u, i, d, spec_doc, spec_msk, r)
	# 	# print(spec_pos)
	# 	alpha = model.alpha.numpy()
	# 	theta = model.theta.numpy()
	# 	alpha = alpha.reshape((-1, args.his_len))
	# 	# print(alpha - theta)
	# 	pos = np.mean(spec_pos*(alpha - theta))
	# 	neg = np.mean((1 - spec_pos)*(alpha - theta)*spec_msk)
	# 	print(pos, neg)
	# 	if b == 10:
	# 		break
	print("============= test data ====================")
	data_loader.reset_test_pt()
	total_batch = int(data_loader.test_size/args.batch_size)+1
	dic = {}
	for b in range(total_batch):
		r, u, i, d, spec_doc, spec_msk, spec_pos, spec_idx = data_loader.test_next_batch()
		model.get_rate_loss(u, i, d, spec_doc, spec_msk, r)
		alpha = model.alpha.numpy()
		theta = model.theta.numpy()
		alpha = alpha.reshape((-1, args.his_len))
		process_alpha(u,i,alpha, theta, spec_pos, spec_msk, spec_idx, dic)
		# print(alpha - theta)

		# if b == 100:
		# 	break
	pickle.dump(dic, open('./images/debug.pkl', 'wb'))

	keys = [key for key in dic.keys()]
	keys = np.sort(keys)[::-1]
	top_k = 3
	for key in keys[:top_k]:
		item = dic[key]
		u,i,alpha,theta,spec_pos, spec_msk, spec_idx, dif = item
		print(alpha)
		print(theta)
		print(spec_pos)
		print(spec_msk)
		print(dif)
		print('\n')

def process_alpha(u,k, alpha, theta, spec_pos, spec_msk, spec_idx, dic):
	for i in range(len(alpha)):
		pos = spec_msk[i]*spec_pos[i]*(alpha[i] - theta[i])
		pos_num = len(np.where(pos>0)[0])
		pos_ratio = len(np.where(pos>0)[0])*1.0/max(1,np.sum(spec_pos[i]))

		neg = spec_msk[i]*(1-spec_pos[i])*(theta[i] - alpha[i])
		neg_num = len(np.where(neg>0)[0])*1.0
		neg_ratio = len(np.where(neg>0)[0])*1.0/max(1,np.sum(spec_msk[i]*(1-spec_pos[i])))
		# neg = spec_msk[i]*(1 - spec_pos[i])*(alpha[i] - theta[i])
		dif = (pos_num+neg_num)/np.sum(spec_msk[i])
		# dif = len(np.where(pos>0)[0])*1.0/max(1,np.sum(spec_pos[i]))
		if np.sum(spec_pos[i])/np.sum(spec_msk[i])<=0.6 and np.sum(spec_pos[i])/np.sum(spec_msk[i])>=0.4:
			dic[dif] = [u[i],k[i],alpha[i], theta[i], spec_pos[i], spec_msk[i], spec_idx[i], dif]
		# print(np.mean(pos), np.mean(neg))

def train(model, data_loader, args):
	mse = 25
	path = './ckpt/'+args.dataset+'/'+"tradeoff_"+str(args.tradeoff)+'_scalar_'+str(args.scalar)+'_hislen_'+str(args.his_len)+'/'
	# path = './ckpt/'+args.dataset+'/'+"ratio_"+str(args.ratio)+'_num_'+str(args.num)+'/'
	if not os.path.exists(path):
		os.makedirs(path)
	fr = open(path+'/result.txt', 'a')


	optimizer   = tf.keras.optimizers.Adam(args.lr)

	for k in range(args.epoch):

		# total_batch = int(data_loader.data_size/args.batch_size)+1
		# data_loader.reset_train_pt()
		# for b in range(total_batch):
		# 	ui, vj, doc, rand_doc = data_loader.dis_next_batch()
		# 	with tf.GradientTape(persistent = True) as tape:
		# 		# model.get_rate_loss(u,i,d,u_his,u_mask,i_his,i_mask,d_rand,r)
		# 		model.get_dis_loss(ui, vj, doc, rand_doc)
		# 		loss = model.dis_loss
		# 	gradients = tape.gradient(loss, model.get_dis_variables())
		# 	optimizer.apply_gradients(zip(gradients, model.get_dis_variables()))
		# 	del tape 
		# 	sys.stdout.write('\repoch:{}/{}, batch:{}/{}, loss={}'.format(k, args.epoch, b, total_batch, loss.numpy()))
		# 	sys.stdout.flush()



		total_batch = int(data_loader.train_size/args.batch_size)+1
		data_loader.reset_train_pt()
		for b in range(total_batch):
			r, u, i, d, spec_doc, spec_msk, spec_pos = data_loader.next_batch()
			with tf.GradientTape(persistent = True) as tape:
				model.get_rate_loss(u, i, d, spec_doc, spec_msk, r)
				model.get_dis_loss(spec_pos, spec_msk)
				loss = model.get_total_loss()
				# loss = model.mse_loss
			gradients = tape.gradient(loss, model.trainable_variables)
			optimizer.apply_gradients(zip(gradients, model.trainable_variables))
			del tape 
			sys.stdout.write('\repoch:{}/{}, batch:{}/{}, loss={}'.format(k, args.epoch, b, total_batch, loss.numpy()))
			sys.stdout.flush()

			if((k*total_batch+b+1)%args.check_freq == 0):
				a_mse, a_mae = val(model, data_loader, args)
				print('\n',a_mse, a_mae,'\n')
				fr.write(str(loss.numpy())+'\t'+str(a_mse)+'\t'+str(a_mae)+'\n')
				if a_mse < mse:
					mse = a_mse
					ckpt = tf.train.Checkpoint(Model = model)
					ckpt.save(path+'/model.ckpt')


def val(model, data_loader, args):
	MAE = tf.keras.metrics.MeanAbsoluteError()
	MSE = tf.keras.metrics.MeanSquaredError()

	data_loader.reset_test_pt()
	total_batch = int(data_loader.test_size/args.batch_size)+1
	for b in range(total_batch):
		r,u,i,d, spec_doc, spec_msk, spec_pos, spec_idx = data_loader.test_next_batch()
		model.get_rate_loss(u, i, d, spec_doc, spec_msk, r)
		rij = model.ratings # (batch_size, 1)
		MSE.update_state(rij, r.reshape((-1,1)))
		MAE.update_state(rij, r.reshape((-1,1)))
	return MSE.result().numpy(), MAE.result().numpy()
	# r,u,i,d,u_his,u_mask, i_his, i_mask, d_rand, spec_doc = data_loader.next_batch()
	# print(r.shape, u.shape, i.shape, d.shape, u_his.shape, u_mask.shape, i_his.shape, i_mask.shape, d_rand.shape, spec_doc.shape)



if __name__ == '__main__':


	parser = argparse.ArgumentParser()
	parser.add_argument('-his_len',    type=int, default=30)
	parser.add_argument('-doc_size',   type=int, default=256)
	parser.add_argument('-batch_size', type=int, default=64)
	parser.add_argument('-dataset',    type=str, default="Office_Products")
	parser.add_argument('-embed_size', type=int, default=64)
	parser.add_argument('-latent_size',type=int, default=64)
	parser.add_argument('-scalar',     type=float, default=5.0)
	parser.add_argument('-tradeoff',   type=float, default=0.5)
	parser.add_argument('-mode',       type=str, default='train')
	parser.add_argument('-epoch',      type=int, default=5)
	parser.add_argument('-lr',         type=float, default=1e-4)
	parser.add_argument('-check_freq',      type=int, default=50)
	parser.add_argument('-ratio',      type=float, default=0.5)
	parser.add_argument('-num',        type=int,   default=10)

	args = parser.parse_args()

	data_loader = Data_Loader(args)

	parser.add_argument('-num_user', type=int, default=data_loader.num_user)
	parser.add_argument('-num_item', type=int, default=data_loader.num_item)
	args = parser.parse_args()

	model = Model(args)

	if args.mode == 'train':
		train(model, data_loader, args)
	elif args.mode == 'debug':
		debug(model, data_loader, args)