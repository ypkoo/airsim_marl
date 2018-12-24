import setup_path

import numpy as np
import sys

sys.path.insert(0, "C:/Users/lanada/Desktop/AirSim/PythonClient")
import airsim



class Agent():
	def __init__(self, name):

		self.name = name


class World(object):
	def __init__(self, agent_n):

		self.airsim_client = MultiAgentClient(agent_n)
		self.agent_n = agent_n
		self.action_dim = 9
		self.size = 100

class MultiAgentClient(airsim.MultirotorClient):

	def __init__(self, agent_n):
		super(airsim.MultirotorClient, self).__init__()
		# agent init
		self.agent_n = agent_n
		self.agent_names = []
		for i in range(self.agent_n):
			name = "Drone" + str(i+1)
			self.agent_names.append(name)

		self.velocity = 5
		self.duration = 10

	def take_action(self, action, agent_index):
		# stop
		if action == 9:
			self.moveByVelocityAsync(0, 0, 0, duration=self.duration, vehicle_name=self.agent_names[agent_index])
		# move
		else:
			vx = self.velocity * np.cos(np.radians(action * 30))
			vy = self.velocity * np.sin(np.radians(action * 30))
			vz = -1.3
			self.moveByVelocityZAsync(vx, vy, vz, duration=self.duration, vehicle_name=self.agent_names[agent_index])

	def take_off_all(self):

		for agent in self.agent_names:
			self.takeoffAsync(timeout_sec = 4, vehicle_name=agent)
			self.moveToPositionAsync(0,0,-1.3,2, vehicle_name=agent)
			self.hoverAsync(vehicle_name=agent)


	def reset_all(self):
		self.reset()
		for agent in self.agent_names:
			self.armDisarm(True, vehicle_name=agent)
			self.enableApiControl(True, vehicle_name=agent)
		self.take_off_all()



