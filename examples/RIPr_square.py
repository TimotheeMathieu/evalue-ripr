import numpy as np
import matplotlib.pyplot as plt
from evalue_ripr import RIP_FW_multi

from scipy.optimize import LinearConstraint

d = 2

# Define constraint of H_0 as the square min(x,y) >= 1/2
A = np.eye(d)
mycons = LinearConstraint(A, 1/2, 1)

# Initialization in this constraint set: it should not be on boundary
init = 0.6*np.ones(d)

# Define point I want to project
q = np.array([0.2, 0.6])

p, kinf = RIP_FW_multi([mycons], [init], 
                        {"prior_support":[q], "prior_weights":[1]}, # We project a mixture proba with only one point in the support
                        use_posterior=False)


fig, ax = plt.subplots()

# Plot constraint set
rect = plt.Rectangle((0.5, 0.5), 0.5, 0.5,  edgecolor=None)
ax.add_patch(rect)

# Plot the point and projection
ax.scatter([q[0]],[q[1]], color="blue", label="point to project")

# Plot the support of the prior obtained as projection.
ax.scatter([pp[0] for pp in p["prior_support"]],
           [pp[1] for pp in p["prior_support"]], color="red", label="projection")
# In this particular case, turns out there is only one support point because we project on convex set

ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')

plt.show()
