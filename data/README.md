Dataset Overview
This dataset contains agent-level observations collected from simulated aerial 
swarm scenarios under varying swarm sizes, obstacle conf igurations, and attrition 
conditions.
Each row corresponds to an individual agent at a particular simulation timestep 
and contains information about the simulation scenario, environment, agent 
kinematics, swarm geometry, neighboring agents, graph-based connectivity, 
and agent criticality.
The dataset is intended for analyzing and predicting the criticality of agents 
within an aerial swarm under dif ferent environmental and attrition conditions.
￼
Variable Descriptions
Scenario and Environment Variables
case_id Unique identif ier for each simulation scenario. Example: c001_n60_none 
indicates case #1, a swarm with 60 agents, and no attrition.
num_agents Total number of agents in the swarm (critical + decoy) at the 
beginning of the scenario.
n_obstacles Number of obstacles present in the environment. Obstacles can be 
represented as spheres, cylinders, or boxes. The number of obstacles varies 
across scenarios.
obstacle_density Fraction of the 200 × 200 simulation area covered by obstacles. 
Values range from 0 to 1, where 0 represents no obstacle coverage and larger 
values indicate greater obstacle coverage.
mean_obstacle_size Average radius or half-width of the obstacles present in the 
environment.
combines information about obstacle density and obstacle count/size so that 
environments containing either a few large obstacles or many smaller obstacles 
can be represented as highly cluttered.
A value near 0 represents an open environment with little or no obstruction, while
higher values indicate increasingly cluttered or obstructed environments.
obstacle_clustering Describes how spatially concentrated the obstacles are.
￼
0 indicates that obstacles are distributed relatively independently 
throughout the environment.
1 indicates that obstacles are strongly concentrated into a small number 
of regions, such as clusters of buildings in an urban environment.
Simulation and Agent Identif ication
step Simulation timestep at which the observation was recorded. The dataset is 
sampled every second simulation step (sample_every = 2).
agent_id Unique index identifying an individual agent within a given simulation 
scenario.
￼
Agent Position and Distance Variables
rel_x, rel_y, rel_z Agent position relative to the swarm centroid along the x, y, and z 
axes.
dist_to_centroid Euclidean distance between the agent and the swarm centroid.
rank_dist_to_centroid Percentile rank of the agent's distance to the swarm centroid 
among the agents alive at that timestep.
0 indicates the closest agent.
1 indicates the farthest agent.
dist_to_target Euclidean distance between the agent and the mission target point.
agents at that timestep.
￼
Agent Kinematic Variables
vx, vy, vz Velocity components of the agent along the x, y, and z axes.
speed Scalar speed of the agent according to the Dubins-model state.
heading Agent yaw angle, describing its horizontal direction of travel.
f light_path_angle Agent f light-path angle (pitch/climb angle), describing its 
vertical direction of travel.
accel Along-track acceleration command applied to the agent during the 
previous simulation step.
heading_rate Yaw-rate command applied to the agent during the previous 
simulation step.
￼
Neighbor and Local Swarm Geometry Variables
num_neighbors Number of other agents located within the agent's sensing range. 
The sensing range is 55 units.
mean_neighbor_dist Mean Euclidean distance between the agent and its 
neighboring agents within the sensing range.
min_neighbor_dist Distance from the agent to its closest neighboring agent.
weighted_density Distance-weighted measure of local neighbor density. Nearby 
agents contribute more strongly than agents farther away.
angular_coverage Fraction of angular sectors surrounding the agent that contain 
neighboring agents. The angular space is divided into 12 azimuth sectors and 4 
elevation bins. This feature measures how well-surrounded or spatially escorted 
an agent is by other swarm members.
￼
Criticality Variables
criticality_score A continuous score between 0 and 1 representing the estimated 
criticality of an agent based on the geometry and state of the swarm. It 
provides a continuous measure of how critical or high-value an agent appears 
to be within the swarm.
This score is distinct from is_critical, which is the ground-truth binary label.
is_critical Ground-truth binary label indicating whether the agent is an actual 
critical (high-value) asset or a decoy.
1 = critical agent
0 = decoy agent
￼
Graph-Based Centrality Variables
The following variables are standard graph-centrality measures computed from 
the instantaneous neighbor graph of the swarm using NetworkX. They describe 
the structural position and importance of an agent within the swarm's 
connectivity graph at that timestep.
degree_centrality Measures the relative connectivity of an agent based on the 
number of neighboring agents it is directly connected to.
closeness_centrality Measures how close an agent is, in terms of graph distance, to 
the other agents in the swarm connectivity graph.
betweenness_centrality Measures how frequently an agent lies on the shortest paths 
between other agents in the swarm graph, indicating its potential role as a 
connectivity bridge.
eigenvector_centrality Measures an agent's connectivity while also considering the 
importance/connectivity of the agents to which it is connected.
￼
Attrition Conditions
The dataset contains dif ferent levels of decoy-agent attrition.
None No drones are lost during the simulation.
Light 1–2 decoy drones are lost, with losses occurring over 1–2 timesteps.
Medium 2–3 decoy drones are lost, with losses occurring over 2–3 timesteps.
Heavy 3–4 decoy drones are lost, with losses occurring over 3–4 timesteps.
Attrition af fects the number of agents present in the swarm over time, and 
features such as neighbor counts, distances, density, angular coverage, and 
graph centrality are therefore computed based on the agents present/alive at 
each timestep.
