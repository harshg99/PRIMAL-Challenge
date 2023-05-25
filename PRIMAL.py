import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pdb import set_trace as bp
class PRIMALNet(nn.Module):
    '''Implements the base network for actor and critic networks'''

    def __init__(self, conv_layer_sizes=[128,256], goal_layer_size = 12, hidden_layer_size = 512, action_size = 5):
        '''
            conv_layer_sizes: list of sizes of the convolutional layers
            goal_layer_size: size of the goal layer
            hidden_layer_size: size of the hidden layer
            action_size: size of the action space
        '''
        super(PRIMALNet, self).__init__()
        
        self.vggnet =  nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels = conv_layer_sizes[0], kernel_size = 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels = conv_layer_sizes[0], out_channels = conv_layer_sizes[0], kernel_size = 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels = conv_layer_sizes[0], out_channels = conv_layer_sizes[0], kernel_size = 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            nn.Conv2d(in_channels = conv_layer_sizes[0] , out_channels = conv_layer_sizes[1], kernel_size = 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels = conv_layer_sizes[1], out_channels = conv_layer_sizes[1], kernel_size = 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels = conv_layer_sizes[1], out_channels = conv_layer_sizes[1], kernel_size = 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            nn.Conv2d(in_channels = conv_layer_sizes[1], out_channels = hidden_layer_size - goal_layer_size, kernel_size = 2, padding=0)
        )

        self.goal_encoder = nn.Sequential(
            nn.Linear(in_features = 2, out_features = goal_layer_size),
            nn.ReLU()
        )

        self.lstm_in_encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features = hidden_layer_size, out_features = hidden_layer_size),
            nn.ReLU(),
            nn.Linear(in_features = hidden_layer_size, out_features = hidden_layer_size),
        )

        self.lstm = nn.LSTM(input_size = hidden_layer_size, hidden_size = hidden_layer_size, num_layers = 1, batch_first = True)

        self.actor = nn.Sequential(
            nn.Linear(in_features = hidden_layer_size, out_features = action_size),
            nn.Softmax(dim = -1)
        )

        self.critic = nn.Sequential(
            nn.Linear(in_features = hidden_layer_size, out_features = 1)
        )

        self.blocking = nn.Sequential(
            nn.Linear(in_features = hidden_layer_size, out_features = 1),
            nn.Sigmoid()
        )

    def rearrange_3D(self, image, forward = True):
        '''
            image: (batch_size, 10, 10, 4)
        '''
        if forward:
            image = image.permute(0,3,1,2)
        else:
            image = image.permute(0,2,3,1)
        return image

    def forward(self, x):
        '''
            x: tuple(input image (batch_size, 4, 10, 10), goal (batch_size, 2))
        '''
        image, goal = x

        B,S,H,W,C = image.shape
        image = image.reshape(B*S,H,W,C)
        image = self.rearrange_3D(image)

        conv_enc = self.vggnet(image)
        conv_enc = conv_enc.reshape(B,S,-1)
        goal_enc = self.goal_encoder(goal)
        #conv_enc = self.rearrange_3D(image, forward = False).squeeze(dim = -1)

        print(conv_enc.shape, goal_enc.shape)

        lstm_in = torch.cat((conv_enc, goal_enc), dim = -1)
        lstm_in = self.lstm_in_encoder(lstm_in)

        lstm_out, _ = self.lstm(lstm_in)

        policy = self.actor(lstm_out)
        value = self.critic(lstm_out)
        blocking = self.blocking(lstm_out)

        return policy, value, blocking


class PRIMAL:
    '''
        THis class implements the PRIMAL algorithm as mentioned in the paper https://arxiv.org/pdf/1809.03531.pdf
    '''
    def __init__(self, gamma = 0.99):
        
        self.network = PRIMALNet()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = 0.0001)
        self.gamma = 0.99
    
    def get_action(self, state, goal):
        
        policy, _, _ = self.network((state, goal))
    
    def backward(self, states, next_states, rewards, actions, dones, blocking_gt):
        
        policy, value, blocking = self.network(states)
        _, next_value, _ = self.network(next_states)

        policy = policy.gather(1, actions)
        value = value.squeeze(dim = -1)
        blocking = blocking.squeeze(dim = -1)

        value_targets =  next_value.detach()*self.gamma + rewards
        value_loss = F.mse_loss(value, value_targets)
        actor_loss = -torch.log(policy)*(value_targets.detach() - value.detach())
        blocking_loss = F.mse_loss(blocking, blocking_gt)

        self.optimizer.zero_grad()
        loss = value_loss + actor_loss + blocking_loss
        loss.backward()
        self.optimizer.step()


    def save(self, path):
        torch.save(self.network.state_dict(), path)
    
    def load(self, path):
        self.network.load_state_dict(torch.load(path))


'''Tests the forward inference of the network,
    We dont test the PRIMAL algorithm as we don't have an environment to collect data from 
'''

def generate_random_map(size = 30):
    
    occupancy_map = np.random.randint(0, 2, size = (size, size))

    # Plot map
    plt.imshow(occupancy_map)
    #plt.show()
    return occupancy_map

def tests(size = 30, robot_location = (15,15), fov = 10, seq_length = 10):
    
    occupancy_map  = generate_random_map(size = size)
    
    robot_location = np.array([15,15]) # at the centre of the map

    robot_obs_field = (robot_location[0] - int(fov/2), robot_location[0]+ int(fov/2),
                        robot_location[1] - int(fov/2), robot_location[1]+ int(fov/2))
    
    robot_observations = occupancy_map[robot_obs_field[0]:robot_obs_field[1], robot_obs_field[2]:robot_obs_field[3]]
    
    # Repeating the robot observations as the input is a 4 channel image
    robot_observations = np.repeat(robot_observations[:, :, np.newaxis], 4, axis=-1)

    robot_observations = np.repeat(robot_observations[np.newaxis,np.newaxis, :, :, :], repeats=seq_length, axis=1)
    robot_goals = np.repeat(robot_location[np.newaxis,np.newaxis, :],repeats=seq_length, axis=1)
    

    # Test the forward pass of the PRIMAL network
    primal = PRIMAL()
    policy, value, blocking = primal.network((torch.from_numpy(robot_observations).float(), torch.from_numpy(robot_goals).float()))
    print(policy.shape, value.shape, blocking.shape)
    
    policy_entropy = -torch.sum(policy*torch.log(policy)).detach().numpy().item()

    assert abs(policy_entropy + seq_length*np.log(0.2)) < 0.01, "Policy output is not a random policy" 


if __name__ == "__main__":
    tests()