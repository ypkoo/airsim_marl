import gym
import numpy as np
from make_env import make_env
import sys, random, time
from collections import deque
import random
import tensorflow as tf

ENV_NAME = "MultiagentOccupy-v0"
sys.path.insert(0, "C:/Users/lanada/Desktop/AirSim/PythonClient")
sys.path.insert(0, "C:/Users/lanada2/Desktop/AirSim/PythonClient")

training_step = 100000
epsilon_dec = 1.0/training_step
epsilon_min = 0.1
m_size = 32
b_size = 10000
pre_train_step = m_size * 100
hidden_size = 32
gamma = 0.99
learning_rate = 0.0005
algorithm = "idqn"

class ReplayBuffer:
	def __init__(self):
		self.replay_memory_capacity = b_size  # capacity of experience replay memory
		self.minibatch_size = m_size  # size of minibatch from experience replay memory for updates
		self.replay_memory = deque(maxlen=self.replay_memory_capacity)

	def add_to_memory(self, experience):
		self.replay_memory.append(experience)

	def sample_from_memory(self):
		return random.sample(self.replay_memory, self.minibatch_size)

	def erase(self):
		self.replay_memory.popleft()

class DQNetwork(object):
	def __init__(self, sess, state_dim, action_dim_single, n_agent):
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim_single = action_dim_single
		self.n_agent = n_agent
		self.action_dim = action_dim_single * n_agent
		# placeholders
		self.s_in = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
		self.a_in = tf.placeholder(dtype=tf.float32, shape=[None, self.n_agent, self.action_dim_single])

		if algorithm == "vdn":
			self.y_in = tf.placeholder(dtype=tf.float32, shape=[None,1])
			with tf.variable_scope('q_network'):
				self.q_network, self.actor_network = self.generate_VDN(self.s_in, self.a_in, True)
			with tf.variable_scope('target_q_network'):
				self.target_q_network, self.target_actor_network = self.generate_VDN(self.s_in, self.a_in, False)

		elif algorithm == "idqn":
			self.y_in = tf.placeholder(dtype=tf.float32, shape=[None,self.n_agent])
			with tf.variable_scope('q_network'):
				self.q_network, self.actor_network = self.generate_IDQN(self.s_in, self.a_in, True)
			with tf.variable_scope('target_q_network'):
				self.target_q_network, self.target_actor_network = self.generate_IDQN(self.s_in, self.a_in, False)

		# indicators (go into target computation)
		self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())  # for dropout
		self.action_onehot = tf.one_hot(self.actor_network, self.action_dim_single, on_value=1.0, off_value=0.0, axis=-1)
		self.target_action_onehot = tf.one_hot(self.target_actor_network, self.action_dim_single, on_value=1.0, off_value=0.0, axis=-1)

		with tf.variable_scope('optimization'):
			self.delta = self.y_in - self.q_network
			self.clipped_error = tf.where(tf.abs(self.delta) < 1.0,
								0.5 * tf.square(self.delta),
								tf.abs(self.delta) - 0.5, name='clipped_error')
			self.cost = tf.reduce_mean(self.clipped_error) 


			self.train_network = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

		o_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_network')
		t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_network')
		self.update_target_network = [tf.assign(t, o) for o, t in zip(o_params, t_params)]

		self.cnt = 0
		self.replay_buffer = ReplayBuffer()
		self.target_update_period = 1000


	def generate_single_q_network(self, obs_single, action_single, reuse, trainable=True):

		hidden_1 = tf.layers.dense(obs_single, hidden_size, activation=tf.nn.relu,
									kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
									bias_initializer=tf.constant_initializer(0.01),  # biases
									use_bias=True,
									trainable=trainable, reuse=reuse, name='dense_h1')
		hidden_2 = tf.layers.dense(hidden_1, hidden_size, activation=tf.nn.relu,
									kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
									bias_initializer=tf.constant_initializer(0.01),  # biases
									use_bias=True,
									trainable=trainable, reuse=reuse, name='dense_h2')
		hidden_3 = tf.layers.dense(hidden_2, hidden_size, activation=tf.nn.relu,
									kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
									bias_initializer=tf.constant_initializer(0.01),  # biases
									use_bias=True,
									trainable=trainable, reuse=reuse, name='dense_h3')
		q_values = tf.layers.dense(hidden_3, self.action_dim_single, trainable=trainable)

		optimal_action = tf.expand_dims(tf.argmax(q_values, 1),-1)
		q = tf.reduce_sum(q_values * action_single, axis=1, keep_dims=True)
		qmax = tf.reduce_max(q_values, axis=1, keep_dims=True)
		return q, optimal_action, qmax

	def generate_VDN(self, s, action, trainable=True):
		q_list = list()
		action_list = list()
		for i in range(self.n_agent):
			# obs = tf.reshape(s, shape=[-1, self.state_dim/2, 2])
			# loc = tf.reshape(obs[:,i,:], shape = [-1, 1, 2])
			# obs = obs - loc
			# obs = tf.reshape(obs, shape=[-1, self.state_dim])
			# # obs = tf.where(tf.abs(obs) < 0.3,
			# #                         obs,
			# #                         obs*0, name='clipped_error')
			# obs_n = tf.concat([obs, tf.reshape(loc, shape = [-1, 2])],1)
			obs = tf.reshape(s, shape=[-1, 14, 2])
			obs_n = tf.concat([obs[:,:,i],obs[:,:,1-i]],1)
			if i == 0:
				reuse = False
			else:
				reuse = True
			q_single, optimal_action, _ = self.generate_single_q_network(obs_n, action[:,i,:], reuse, trainable)
			q_list.append(q_single)
			action_list.append(optimal_action)
		q_values = tf.concat(q_list, axis=1)
		optimal_action = tf.concat(action_list, axis=1)

		q_value = tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True)

		return q_value, optimal_action

	def generate_IDQN(self, s, action, trainable=True):
		q_list = list()
		action_list = list()
		for i in range(self.n_agent):
			# obs = tf.reshape(s, shape=[-1, self.state_dim/2, 2])
			# loc = tf.reshape(obs[:,i,:], shape = [-1, 1, 2])
			# obs = obs - loc
			# obs = tf.reshape(obs, shape=[-1, self.state_dim])
			# # obs = tf.where(tf.abs(obs) < 0.3,
			# #                         obs,
			# #                         obs*0, name='clipped_error')
			# obs_n = tf.concat([obs, tf.reshape(loc, shape = [-1, 2])],1)
			obs = tf.reshape(s, shape=[-1, 14, 2])
			obs_n = tf.concat([obs[:,:,i],obs[:,:,1-i]],1)
			if i == 0:
				reuse = False
			else:
				reuse = True
			q_single, optimal_action, _ = self.generate_single_q_network(obs_n, action[:,i,:], reuse, trainable)
			q_list.append(q_single)
			action_list.append(optimal_action)
		q_values = tf.concat(q_list, axis=1)
		optimal_action = tf.concat(action_list, axis=1)

		# q_value = tf.reduce_sum(q_values, reduction_indices=1, keep_dims = True)

		return q_values, optimal_action

	def get_action(self, state_ph):
		state_ph = state_ph[None]

		return self.sess.run(self.actor_network, feed_dict={self.s_in: state_ph})[0]

	def get_q_values(self, state_ph, action_ph):
		return self.sess.run(self.q_network, feed_dict={self.s_in: state_ph,
														self.a_in: action_ph})

	def get_target_q_values(self, state_ph, action_ph):
		return self.sess.run(self.target_q_network, feed_dict={self.s_in: state_ph,
													self.a_in: action_ph})
	    


	def training_qnet(self, minibatch):
		y = []
		self.cnt += 1


		# Get target value from target network
		target_action = self.sess.run(self.target_action_onehot, 
			feed_dict={self.s_in: [data[3] for data in minibatch]})
		target_q_values = self.sess.run(self.target_q_network, 
			feed_dict={self.s_in: [data[3] for data in minibatch], self.a_in: target_action})
		# target_q_values = self.sess.run(self.target_v, feed_dict={self.s_in: [data[3] for data in minibatch]})
		y = np.zeros([m_size])
		r = np.array([[data[2]] for data in minibatch]).reshape(m_size,-1)
		done = np.array([[data[4]] for data in minibatch])
		y = r + gamma * (1-done) * target_q_values
		# c = self.sess.run(self.cost, feed_dict={self.y_in: y,
		#                             self.a_in: [data[1] for data in minibatch],
		#                             self.s_in: [data[0] for data in minibatch]
		#                             })
		self.sess.run(self.train_network, feed_dict={
						self.y_in: y,
						self.a_in: [data[1] for data in minibatch],
						self.s_in: [data[0] for data in minibatch]
				})

	def training_target_qnet(self):
		"""
		copy weights from q_network to target q_network
		:return:
		"""
		self.sess.run(self.update_target_network)


	def action_to_onehot(self, action):
		onehot = np.zeros([self.n_agent, self.action_dim_single])
		for i in range(self.n_agent):
			onehot[i, action[i]] = 1
		return onehot

	def train_agents(self, state, action, reward, state_n, done):


		s = state
		a = self.action_to_onehot(action)
		r = reward
		s_n = state_n
		# r = np.sum(reward)
		self.replay_buffer.add_to_memory((s,a,r,s_n,done))

		if len(self.replay_buffer.replay_memory) < pre_train_step:
			return 0

		minibatch = self.replay_buffer.sample_from_memory()
		self.training_qnet(minibatch)
		self.cnt += 1

		if self.cnt % self.target_update_period == 0:
			self.training_target_qnet()
   			






def main():
	# env = gym.make(ENV_NAME)
	env = make_env('occupy')
	
	state_dim = 14 * env.agent_n # TODO: get from env
	action_dim_single = 9 # TODO: get from env
	n_agent = env.agent_n
	train = True
	tf.reset_default_graph()
	my_graph = tf.Graph()
	with my_graph.as_default():
		sess = tf.Session(graph=my_graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
		q_network = DQNetwork(sess, state_dim, action_dim_single, n_agent)
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()

	run = 0
	step = 0
	while True:
		step_n = 0
		state = env.reset()
		state = np.array(state[0])
		reward_sum = 0

		while True:

			start_time = time.time()
			action_n = []
			epsilon = max(epsilon_min, 1.0 - epsilon_dec * step) 
			action_list = q_network.get_action(state)
			for i in range(env.agent_n):
				if train and (step < pre_train_step or np.random.rand() < epsilon):
					action_n.append(random.randrange(0, 9))
				else:
					action_n.append(action_list[i])

			state_n, reward_n, done_n, info_n = env.step(action_n)
			state_n = np.array(state_n[0])
			done_single = sum(done_n) > 0
			if algorithm == "vdn": 
				reward = reward_n[0]
			else:
				reward = np.array(reward_n)
			reward_sum += reward
			q_network.train_agents(state, action_n, reward, state_n, done_single)
			state = state_n
			# print("run: %d, step: %d, step_n: %d, reward: %f" % (run, step, step_n, reward_n[0]))

			if step % 10000 == 0 and step < 140000:
				saver.save(sess, "C:\\Users\lanada2\Desktop\AirSim\weights\weights-"+str(step).rjust(7,'0'))
			if any(done_n) or step_n >= 100:
				break

			step_n = step_n + 1
			step = step + 1

			end_time = time.time()
			time.sleep(0.25 - (end_time - start_time))
		run = run + 1
		print("run: %d, step: %d, step_n: %d, " % (run, step, step_n), "reward: ", reward_sum)

if __name__ == "__main__":
	main()