class Model(object):
	def __init__(self, BATCH_SIZE, INPUT_H, INPUT_W, is_training):
		self.batchsize = BATCH_SIZE
		self.G = Generator('Generator')
		self.var_list = []

		self.step = 0
		if not hasattr(self, "ewc_loss"):
			self.Add_loss = 0

		self.SOURCE1 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, INPUT_H, INPUT_W, 1), name = 'SOURCE1')
		self.SOURCE2 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, INPUT_H, INPUT_W, 1), name = 'SOURCE2')

		self.c = tf.placeholder(tf.float32, shape = (), name = 'c')
		print('source shape:', self.SOURCE1.shape)

		self.generated_img = self.G.transform(I1 = self.SOURCE1, I2 = self.SOURCE2, is_training = is_training, reuse=False)
		self.var_list.extend(tf.trainable_variables())

		# for i in self.var_list:
		# 	print(i.name)

		if is_training:
			''' SSIM loss'''
			SSIM1 = 1 - SSIM_LOSS(self.SOURCE1, self.generated_img)
			SSIM2 = 1 - SSIM_LOSS(self.SOURCE2, self.generated_img)
			mse1 = Fro_LOSS(self.generated_img-self.SOURCE1)
			mse2 = Fro_LOSS(self.generated_img-self.SOURCE2)

			with tf.device('/gpu:1'):
				self.S1_VGG_in = tf.image.resize_nearest_neighbor(self.SOURCE1, size = [224, 224])
				self.S1_VGG_in = tf.concat((self.S1_VGG_in, self.S1_VGG_in, self.S1_VGG_in), axis = -1)
				self.S2_VGG_in = tf.image.resize_nearest_neighbor(self.SOURCE2, size = [224, 224])
				self.S2_VGG_in = tf.concat((self.S2_VGG_in, self.S2_VGG_in, self.S2_VGG_in), axis = -1)

				vgg1 = Vgg16()
				with tf.name_scope("vgg1"):
					self.S1_FEAS = vgg1.build(self.S1_VGG_in)
				vgg2 = Vgg16()
				with tf.name_scope("vgg2"):
					self.S2_FEAS = vgg2.build(self.S2_VGG_in)

				for i in range(len(self.S1_FEAS)):
					self.m1 = tf.reduce_mean(tf.square(features_grad(self.S1_FEAS[i])), axis = [1, 2, 3])
					self.m2 = tf.reduce_mean(tf.square(features_grad(self.S2_FEAS[i])), axis = [1, 2, 3])
					if i == 0:
						self.ws1 = tf.expand_dims(self.m1, axis = -1)
						self.ws2 = tf.expand_dims(self.m2, axis = -1)
					else:
						self.ws1 = tf.concat([self.ws1, tf.expand_dims(self.m1, axis = -1)], axis = -1)
						self.ws2 = tf.concat([self.ws2, tf.expand_dims(self.m2, axis = -1)], axis = -1)

			self.s1 = tf.reduce_mean(self.ws1, axis = -1) / self.c
			self.s2 = tf.reduce_mean(self.ws2, axis = -1) / self.c
			self.s = tf.nn.softmax(
				tf.concat([tf.expand_dims(self.s1, axis = -1), tf.expand_dims(self.s2, axis = -1)], axis = -1))


			self.ssim_loss = tf.reduce_mean(self.s[:, 0] * SSIM1 + self.s[:, 1] * SSIM2)
			self.mse_loss = tf.reduce_mean(self.s[:, 0] * mse1 + self.s[:, 1] * mse2)
			self.content_loss = self.ssim_loss + 20 * self.mse_loss