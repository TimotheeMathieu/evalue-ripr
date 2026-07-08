"""
In this example, we use the evalue_ripr package to project, in 2D, a distribution Ber(0.6) \otimes Ber(0.6) on product of Bernoulli distributions with one of its parameter smaller than 1/2
"""

import numpy as np
import matplotlib.pyplot as plt
from evalue_ripr import RIP_FW_multi
from scipy.optimize import LinearConstraint

d = 2

# Define constraint of H_0 as the square min(x,y) < 1/2
# This is two constraints, as this is two convex components
A1 = np.array([0,1])
mycons1 = LinearConstraint(A1, 0, 1/2)

A2 = np.array([1,0])
mycons2 = LinearConstraint(A2, 0, 1/2)

# Initialization in this constraint set: should not be on boundary
inits = [ np.array([0.6, 0.2]), np.array([0.2, 0.6])]

# Define point I want to project
q = np.array([0.6, 0.6])

p, kinf = RIP_FW_multi([mycons1, mycons2], inits, 
                        {"prior_support":[q], "prior_weights":[1]}, # We project a mixture proba with only one point in the support
                        use_posterior=False)

print(p["prior_support"])
fig, ax = plt.subplots()

# Plot constraint set
rect1 = plt.Rectangle((0, 0), 0.5, 0.5,  edgecolor=None)
rect2 = plt.Rectangle((0, 0.5), 0.5, 0.5,  edgecolor=None)
rect3 = plt.Rectangle((0.5, 0), 0.5, 0.5,  edgecolor=None)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)

# Plot the point and projection
ax.scatter([q[0]],[q[1]], color="blue", label="point to project", s = 50)

# Plot the support of the prior obtained as projection. Size proportional to weight.
ax.scatter([pp[0] for pp in p["prior_support"]],
           [pp[1] for pp in p["prior_support"]], color="red", label="projection", s = p["prior_weights"]*50)
# In this particular case, there are multiple points. 
# Remark that there is no unicity of the solution. It is very probable that a simpler support could be found. We just provide one.

ax.legend()
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_aspect('equal')

plt.show()
