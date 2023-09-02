import numpy as np 
import tensorflow as tf
from tensorflow import keras
import argparse
import os


class Model(tf.keras.Model):
	def __init__(self, args):
		super(Model, self).__init__()
		self.args = args

		with tf.name_scope(name='dis_variables'):
			self.W_1        = tf.keras.layers.Dense(self.args.latent_size, use_bias=False)
			self.W_2        = tf.keras.layers.Dense(self.args.latent_size)
			self.nu         = tf.keras.layers.Dense(1, activation='relu')
			self.t          = tf.keras.layers.Dense(1, activation='relu')
		with tf.name_scope(name='rate_variables'):
			self.user_embed = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(self.args.num_user, self.args.embed_size)))
			self.item_embed = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(self.args.num_item, self.args.embed_size)))
			# self.uthe_embed = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(self.args.num_user, 1)))
			# self.tanh       = tf.keras.layers.Activation('tanh')
			self.hidden_layer1 = tf.keras.layers.Dense(int(self.args.latent_size/2), activation='relu')
			self.hidden_layer2 = tf.keras.layers.Dense(int(self.args.latent_size/4), activation='relu')
			self.hidden_layer3 = tf.keras.layers.Dense(1)

		
	def get_dis_variables(self):
		return [self.W_1.trainable_variables[0],\
		self.W_2.trainable_variables[0],\
		self.W_2.trainable_variables[1],\
		self.nu.trainable_variables[0],\
		self.nu.trainable_variables[1],\
		self.t.trainable_variables[0],\
		self.t.trainable_variables[1]]
	def get_rate_variables(self):
		return [self.user_embed,\
		self.item_embed,\
		self.hidden_layer1.trainable_variables[0],\
		self.hidden_layer1.trainable_variables[1],\
		self.hidden_layer2.trainable_variables[0],\
		self.hidden_layer2.trainable_variables[1],\
		self.hidden_layer3.trainable_variables[0],\
		self.hidden_layer3.trainable_variables[1]]

	def get_rate_loss(self, user_ids, item_ids, doc_vec, his_vecs, his_msk, ground_truth):
		user_embed = tf.nn.embedding_lookup(self.user_embed, user_ids) # shape=(batch_size, embed_size)
		item_embed = tf.nn.embedding_lookup(self.item_embed, item_ids) # shape=(batch_size, embed_size)
		# uthe_embed = tf.nn.embedding_lookup(self.uthe_embed, user_ids) # (batch_size,1)
		# ithe_embed = tf.nn.embedding_lookup(self.ithe_embed, item_ids) # (batch_size,1)
		ui_concat  = tf.concat([user_embed, item_embed], axis=-1)

		spec_vec   = self.get_alpha(ui_concat, his_vecs, his_msk, doc_vec)
		# doc_j      = self.get_alpha(ui_concat, his_vecs, his_msk)
		self.ratings    = self.MLP(user_embed, item_embed, spec_vec) # (batch_size,1)
		
		mse_loss   = tf.keras.losses.MeanSquaredError()(self.ratings, ground_truth.reshape((-1,1)))
		# c_loss     = self.contrastive(ui_concat, doc_vec, doc_rand_vec, uthe_embed) + self.contrastive(ui_concat, doc_vec, doc_rand_vec, ithe_embed)

		self.mse_loss  = mse_loss

	def get_alpha(self,ui_concat, his_vecs, his_mask, doc_vec):
		#ui_concate: (batch_size, 2*embed_size)
		#his_vecs:(batch_size, his_len, 256)
		#his_mask:(batch_size, his_len)
		#theta: (batch_size, 1)
		# self.theta = self.t(ui_concat) # (batch_size, 1)
		# self.get_beta(doc_vec, his_vecs, his_mask) # self.beta = (batch_size, his_len)

		self.theta= self.t(ui_concat) # (batch_size, 1)
		ui_expand = tf.expand_dims(doc_vec, axis=1) # (batch_size, 1, 2*embed_size)
		his_vecs_t   = self.W_2(his_vecs) # (batch_size, his_len, latent_size)
		self.alpha        = self.nu(tf.math.tanh(self.W_2(ui_expand)+his_vecs_t)) # (batch_size, his_len, 1)
		alpha_reduce = tf.reshape(self.alpha, shape = (-1, self.args.his_len)) # (batch_size, his_len)

		alpha_scalar = tf.math.sigmoid(self.args.scalar*(alpha_reduce - self.theta)) #(batch, his_len)
		alpha_mask   = alpha_scalar*his_mask*tf.math.exp(alpha_reduce)
		alpha_sum    = tf.math.reduce_sum(alpha_mask, axis=-1, keepdims=True) # (batch_size, 1)
		self.alpha_norm   = alpha_mask/alpha_sum  # (batch_size, his_len)
		alpha_expand = tf.expand_dims(self.alpha_norm, axis=-1) # (batch_size, his_len, 1)
		his_sum      = tf.math.reduce_sum(alpha_expand*his_vecs_t, axis=1)# (batch_size, 256)
		return his_sum

	def MLP(self, ud_vec, id_vec, spec_vec):
		# ud_vec: (batch_size, latent_size)
		# id_vec: (batch_size, latent_size)
		concat = tf.concat([ud_vec, id_vec, spec_vec], axis=-1) # (batch_size, 2*latent_size)
		hidden_1 = self.hidden_layer1(concat)
		hidden_2 = self.hidden_layer2(hidden_1)
		rating   = self.hidden_layer3(hidden_2)
		return rating

	def get_dis_loss(self, spec_pos, spec_msk):
		# spec_pos: (batch_size, his_len)
		# spec_msk: (batch_size, his_len)
		# theta:    (batch_size, 1)
		# alpha:    (batch_size, his_len, 1)

		alpha_reduce = tf.reshape(self.alpha, shape = (-1, self.args.his_len)) # (batch_size, his_len)
		pos = spec_pos*spec_msk*(alpha_reduce - self.theta) # (batch_size, his_len)
		neg = (1 - spec_pos)*spec_msk*(self.theta - alpha_reduce) #(batch_size, his_len)
		c_loss = -tf.math.log(tf.math.sigmoid(pos)) - tf.math.log(tf.math.sigmoid(neg))
		self.dis_loss = tf.reduce_mean(c_loss)

	def get_total_loss(self):
		# return self.mse_loss+self.args.tradeoff*(tf.cast(self.dis_loss, tf.float64)+0.1*self.get_l2_loss())
		return self.mse_loss+self.args.tradeoff*(tf.cast(self.dis_loss+0.01*self.get_l2_loss(), tf.float64))

	def get_l2_loss(self):
		l2_loss = tf.nn.l2_loss(self.t.trainable_variables[0])
		l2_loss += tf.nn.l2_loss(self.t.trainable_variables[1])

		return l2_loss