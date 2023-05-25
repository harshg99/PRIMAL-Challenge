The primary objective to making PRIMAL amenable to real robot lidar data is to add an intermediate multi-agent mapping module.

We assume this mapping occurs in a centralised setting, where we would take the Lidar data from all n robots in their own respective frames and convert the stream of data
into the world frame. Based on the position of each robot, we can create an empty raster and impose the world frame Lidar data for each robot
on to an empty raster. This would form a multi-agent joint raster scan of the environment at a given time step. Note that any unseen cell will be labelled as -1 in the join raster scan.

From here, we can use particle filter SLAM or any SLAM approach to convert the multi-agent scan raster to a map. Here, the particle filter have 3 modules.

Module 1: Sampling - N random particles sampled from a multi-variate gaussian with a mean corresponding to agent locations
Module 2: Scan matching - computing expected scan rasters from each of the N particles and evaluating the weights of the particles based on the match between the particle rasters
          and the actual rasters
Module 3: Mapping - Updating the multi-variate gaussian based on the particle weights and computing the expected location for each agent based on particle weights. We further
          use the expected location to update the map based on the raster scan obtained from robot lidar data. This can be simply be a log-odds update based on whether a cell 
          is observed.

Once we have this discretised map for the environment in real time, we can now compute the agent centric feature maps as per PRIMAL's requirement. Here the agent
centric feature maps are in a fixed fielf-of-view around the agent and don't consider occlusions. Hence, we require the above mapping module. 
