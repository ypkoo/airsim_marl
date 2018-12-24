import setup_path
import airsim
from airsim_marl.core import MultiAgentClient, World
from airsim_marl.utils import *


class Scenario():
	def make_world(self):
		
		world = World(agent_n=2)

		return world

	def reset_world(self, world):
		world.airsim_client.reset_all()

	def reward(self, agent_index, world):

		goal_a_pos = (1, 2)
		goal_b_pos = (3, 4)

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

		return -(dist_a + dist_b)



	def observation(self, agent_index, world):
		client = world.airsim_client

		pos_list = []

		for i in range(client.agent_n):

			state = client.getMultirotorState(vehicle_name=client.agent_names[i])
			pos = state.kinematics_estimated.position

			pos_list.append(pos)


		return (pos_list[0].x_val, pos_list[0].y_val, pos_list[1].x_val, pos_list[1].y_val)



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