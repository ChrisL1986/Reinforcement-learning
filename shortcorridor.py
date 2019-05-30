#!/usr/bin/python

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import random
import numpy as np

left = -1
right = 1

def step(state, action):
	if state == 1:
		state = state - action
	else:
		state = max(0, state + action)

	return state

def epsilonGreedy(policy, epsilon, episodes):
	v = 0
	for i in range(0, episodes):	
		state = 0
		reward = 0
		while state != 3:
			if random.random() > float(epsilon)/2:
				action = policy
			else:
				action = -policy
			state = step(state, action)
			reward -= 1

		v = v + reward


def reinforceMCgradient(episodes, alpha = 2e-5):
	x_s_right = [1,0]
	x_s_left = [0,1]
	theta = [0.5,0.5]
	for i in range(0, episodes):	
		G = []
		state = 0
		actions = []
		reward = 0
		prob_right = np.exp(np.dot(theta,x_s_right)) / (np.exp(np.dot(theta,x_s_right)) + np.exp(np.dot(theta,x_s_left)))
		while state != 3:
			#go right
			if random.random() < prob_right:
				state = step(state, right)
				actions.append(right)
			#go left
			else:
				state = step(state, left)
				actions.append(left)
			reward -= 1
			G.append(reward)
		G = G[::-1]
		for i in range(0, len(actions)):
			prob_right = np.exp(np.dot(theta,x_s_right)) / (np.exp(np.dot(theta,x_s_right)) + np.exp(np.dot(theta,x_s_left)))
			prob_left = np.exp(np.dot(theta,x_s_left)) / (np.exp(np.dot(theta,x_s_right)) + np.exp(np.dot(theta,x_s_left)))
			if actions[i] == right:
				gradient = np.subtract(x_s_right, prob_right * np.asarray(x_s_right) + prob_left * np.asarray(x_s_left))
			else:
				gradient = np.subtract(x_s_left, prob_right * np.asarray(x_s_right) + prob_left * np.asarray(x_s_left))
			theta = theta + alpha * G[i] * np.asarray(gradient)
		
	prob_right = np.exp(np.dot(theta,x_s_right)) / (np.exp(np.dot(theta,x_s_right)) + np.exp(np.dot(theta,x_s_left)))
	prob_left = np.exp(np.dot(theta,x_s_left)) / (np.exp(np.dot(theta,x_s_right)) + np.exp(np.dot(theta,x_s_left)))



if __name__ == "__main__":
	#epsilonGreedy(1, 0.1, 100000)
	reinforceMCgradient(1000)