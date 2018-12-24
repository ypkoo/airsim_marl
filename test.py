import gym
import numpy as np
from make_env import make_env
import sys, random, time

ENV_NAME = "MultiagentOccupy-v0"
sys.path.insert(0, "C:/Users/lanada/Desktop/AirSim/PythonClient")

def main():
	# env = gym.make(ENV_NAME)
	env = make_env('occupy')
	

	
	run = 0
	while True:
		step_n = 0
		state = env.reset()

		while True:

			action_n = []

			for i in range(env.agent_n):
				action_n.append(random.randrange(1, 10))


			obs_n, reward_n, done_n, info_n = env.step(action_n)

			print("run: %d, step: %d, reward: %d" % (run, step_n, reward_n[0]))

			if any(done_n):
				break

			step_n = step_n + 1

			time.sleep(1)

		run = run + 1


if __name__ == "__main__":
	main()