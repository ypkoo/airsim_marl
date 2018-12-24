import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np

class MultiAgentEnv(gym.Env):
	metadata = {
		'render.modes' : ['human', 'rgb_array']
	}

	def __init__(self, world, reset_callback=None, reward_callback=None,
				 observation_callback=None, info_callback=None,
				 done_callback=None, shared_viewer=True):
		
		self.world = world
		# self.agents = self.world.agents

		self.agent_n = self.world.airsim_client.agent_n

		self.reset_callback = reset_callback
		self.reward_callback = reward_callback
		self.observation_callback = observation_callback
		self.info_callback = info_callback
		self.done_callback = done_callback

		self.discrete_action_space = True

		self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
		self.time = 0



		# action space
		self.action_space = []

		for i in range(self.agent_n):
			if self.discrete_action_space:
				space = spaces.Discrete(world.action_dim)
			else:
				pass

			self.action_space.append(space)

	def step(self, action_n):
		obs_n = []
		reward_n = []
		done_n = []
		info_n = {'n': []}

		# ???
		# self.agents = self.world.agents

		# take an action for each agent
		for agent_index in range(self.agent_n):
			self._take_action(action_n[agent_index], agent_index)

		# self.world.step()

		# observe states
		for agent_index in range(self.agent_n):
			obs_n.append(self._get_obs(agent_index))
			reward_n.append(self._get_reward(agent_index))
			done_n.append(self._get_done(agent_index))

			info_n['n'].append(self._get_info(agent_index))

		reward = np.sum(reward_n)
		if self.shared_reward:
			reward_n = [reward] * self.n

		return obs_n, reward_n, done_n, info_n


	def reset(self):

		self.reset_callback(self.world)

		# self._reset_render()

		obs_n = []

		for agent_index in range(self.agent_n):
			obs_n.append(self._get_obs(agent_index))
		return obs_n

	def _get_info(self, agent_index):
		if self.info_callback is None:
			return {}

		return self.info_callback(agent_index, self.world)


	def _get_obs(self, agent_index):
		if self.observation_callback is None:
			return np.zeros(0)

		return self.observation_callback(agent_index, self.world)

	def _get_done(self, agent_index):
		if self.done_callback is None:
			return False

		return self.done_callback(agent_index, self.world)

	def _get_reward(self, agent_index):
		if self.reward_callback is None:
			return 0.0

		return self.reward_callback(agent_index, self.world)

	def _take_action(self, action, agent_index):
		self.world.airsim_client.take_action(action, agent_index)


