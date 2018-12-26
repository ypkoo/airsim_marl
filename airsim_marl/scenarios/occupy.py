import setup_path
import airsim
from airsim_marl.core import MultiAgentClient, World
from airsim_marl.utils import *
import numpy as np


class Scenario():
	def make_world(self):
		
		world = World(agent_n=2)

		return world

	def reset_world(self, world):
		world.airsim_client.reset_all()

	def reward(self, agent_index, world):

		goal_a_pos = (15, 15)
		goal_b_pos = (-15, -15)

		success_dist = 3

		client = world.airsim_client

		pos_list = []

		for i in range(client.agent_n):

			state = client.getMultirotorState(vehicle_name=client.agent_names[i])
			pos = state.kinematics_estimated.position

			pos_list.append(pos)

		dist_a = min(distance(goal_a_pos, (pos_list[0].x_val, pos_list[0].y_val)), distance(goal_a_pos, (pos_list[1].x_val, pos_list[1].y_val)))
		dist_b = min(distance(goal_b_pos, (pos_list[0].x_val, pos_list[0].y_val)), distance(goal_b_pos, (pos_list[1].x_val, pos_list[1].y_val)))

		

		""" TODO: when drones go out of border """
		reward = 2 -(dist_a + dist_b)/20.
		if reward < 0:
			reward = 0.
		if self.is_out(agent_index, world):
			reward = -10.

		return reward



	def observation(self, agent_index, world):
		client = world.airsim_client

		info_array = np.zeros([14, 2])

		for i in range(client.agent_n):

			info_list = []
			info_list2 = []
			state = client.getMultirotorState(vehicle_name=client.agent_names[i])
			
			info_list.append(state.kinematics_estimated.position)
			info_list.append(state.kinematics_estimated.linear_velocity)
			info_list.append(state.kinematics_estimated.angular_velocity)
			info_list.append(state.kinematics_estimated.linear_acceleration)
			info_list.append(state.kinematics_estimated.angular_acceleration)
			info_list.append(state.kinematics_estimated.orientation)

			for j in range(6):
				info_list2.append(info_list[j].x_val)
				info_list2.append(info_list[j].y_val)
			info_list2.append(info_list[5].w_val)
			info_list2.append(info_list[5].z_val)
			info_array[:,i] = np.array(info_list2)

		# print (info_array)
		return np.reshape(info_array,[28])



	def is_collision(self):
		pass

	def done(self, agent_index, world):
		return self.is_out(agent_index, world)

	def is_out(self, agent_index, world):
		size = world.size
		client = world.airsim_client
		state = client.getMultirotorState(vehicle_name=client.agent_names[agent_index])
		pos = state.kinematics_estimated.position

		return (np.abs(pos.x_val) > size/2) or (np.abs(pos.y_val) > size/2)