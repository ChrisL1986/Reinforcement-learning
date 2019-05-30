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

class Randomwalk:

	def __init__(self, num_states):
		self.current_state = num_states / 2
		self.num_states = num_states + 1
	
	def step(self):
		neighbor = random.randint(1, 100)
		if random.choice([True, False]):
			self.current_state = max(0, self.current_state - neighbor)
		else:
			self.current_state = min(self.num_states, neighbor + self.current_state)

		if self.current_state == 0:
			return 1, -1
		elif self.current_state == self.num_states:
			return 1, 1
		else:
			return 0, 0

	def generate_episode(self):
		episode = []
		episode.append(self.current_state)
		while True:
			end, reward = self.step()
			episode.append(self.current_state)
			if end:
				break
		return episode, reward

def monte_carlo_first_visit(num_states, num_episodes):
	V = [0] * (num_states + 2)
	n = [0] * (num_states + 2)
	for i in range(0, num_episodes):
		r = Randomwalk(num_states)
		episode, reward = r.generate_episode() 
		#remove duplicates using set (first.visit mc)
		episode = list(set(episode))
		for e in episode:
			V[e] += reward
			n[e] += 1
	
	for i in range(1, num_states + 1):
		print(float(V[i])/float(n[i]))

def td_zero(num_states, num_episodes, alpha = 0.01):
	V = [0] * (num_states + 2)
	for i in range(0, num_episodes):
		r = Randomwalk(num_states)
		while True:
			state = r.current_state
			end, reward = r.step()
			V[state] += alpha * (reward + V[r.current_state] - V[state])
			if end:
				break
	for i in range(1, num_states + 1):
		print(V[i])

def n_step_td(num_states, num_episodes, steps, alpha = 0.01):
	V = [0] * (num_states + 2)
	for i in range(0, num_episodes):
		r = Randomwalk(num_states)
		episode, reward = r.generate_episode()
		for t in range(0, len(episode) -1):
			state = episode[t]
			if t + steps >= len(episode) -1:
				G = reward
			else:
				G = 0
			next_n_state = episode[min(len(episode)-1,t + steps)]
			V[state] += alpha * (G + V[next_n_state] - V[state])

	for i in range(1, num_states + 1):
		print(V[i])


def monte_carlo_gradient(num_states, num_episodes, alpha = 0.00002):
	w1 = 0
	w0 = 0
	for i in range(0, num_episodes):
		r = Randomwalk(num_states)
		episode, reward = r.generate_episode() 
		for e in episode:
			x = e/100
			w0_new = w0 + alpha * (reward - (x * w1 + w0))
			w1_new = w1 + alpha * (reward - (x * w1 + w0)) * x
			w0 = w0_new
			w1 = w1_new
	
	for i in range(0, 10):
		print(i, i * w1 + w0)

if __name__ == "__main__":
	#monte_carlo_first_visit(1000, 100000)
	#td_zero(1000, 100000)
	#n_step_td(1000, 200000, 3)
	monte_carlo_gradient(1000, 100000)

