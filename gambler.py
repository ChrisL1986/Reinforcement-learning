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

import matplotlib.pyplot as plt
import numpy as np

CAPITAL = 100
GAMMA = 1
THETA = 1e-9

prob_head = 0.55

#two additional states 0 and CAPITAL
V = np.zeros(CAPITAL + 1)
V[CAPITAL] = 1

#Value Iteration
while 1:
	delta = 0
	for s in xrange(1,CAPITAL):
		a_range = np.minimum(s, CAPITAL - s) + 1
		v = V[s]
		max_new = 0
		for a in xrange(0, a_range):
			V_new = prob_head * GAMMA * V[s + a] + (1 - prob_head) * GAMMA * V[s - a]
			max_new = np.maximum(max_new, V_new)
		V[s] = max_new
		delta = np.maximum(delta, np.absolute(v - V[s]))
	#plt.plot(xrange(0, CAPITAL + 1), V)
	if delta < THETA:
		break

#plt.title('p = %f' % (prob_head))
#plt.ylabel('Value estimates')
#plt.xlabel('Capital')		
#plt.show()

state = []
action = []
#get all optimal policies
for s in xrange(1,CAPITAL):
	a_range = np.minimum(s, CAPITAL - s) + 1
	max = 0
	#get max value 
	for a in xrange(0, a_range):
		max = np.maximum(prob_head * GAMMA * V[s + a] + (1 - prob_head) * GAMMA * V[s - a], max)
	#get all actions with max value
	for a in xrange(0, a_range):
		#error term added (1e-9) because of rounding errors. Otherweise not all optimal policies are selected
		if max <= prob_head * GAMMA * V[s + a] + (1 - prob_head) * GAMMA * V[s - a] + 1e-9:
			state.append(s)
			action.append(a)

#plt.ylim((0,CAPITAL - 1))
#plt.title('p = %f' % (prob_head))
#plt.xlabel('Capital')	
#plt.ylabel('Final policy')
#plt.plot(state, action, 'ro')
#plt.show()