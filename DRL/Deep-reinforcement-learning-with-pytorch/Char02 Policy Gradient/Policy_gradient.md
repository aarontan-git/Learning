# README

## Policy Graident Notes

The policy takes as input the states and returns a probability distribution of the actions to take. 

Define a policy as a neural network with 4 inputs and 2 outputs. The 4 inputs are the states of cartpole environment (cart position, pole position, cart speed, pole speed) and the 2 outputs are the actions (move left, move right).

```
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
```

To select an action
```
def select_action(state): # state = numpy array, state = array([ 0.03184929,  0.00238684, -0.02272496,  0.02353324])
    state = torch.from_numpy(state).float().unsqueeze(0) # converts the numpy array into a tensor
    probs = policy(state) # returns the probability of an action with the given state (eg. output of the NN) = tensor of 1x2
    m = Categorical(probs) # turn the probability of action (categorical = left or right) in to a distribution
    action = m.sample() # sample the action from the distribution computed previously
    policy.saved_log_probs.append(m.log_prob(action)) # log_prob returns the log of the probability density/mass function evaluated at the given sample value
    return action.item()
```