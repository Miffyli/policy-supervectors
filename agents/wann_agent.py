import numpy as np
from gym import spaces

from agents import SimpleAgentClass

# Create agents for the CMA-ES, NEAT and WANN agents
# defined in the weight-agnostic paper repo:
#  https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease/


# -------------------------------------------------------------------
# Here begins copy/paste from WANNRelease code linked above


def weightedRandom(weights):
    """Returns random index, with each choices chance weighted
    Args:
    weights   - (np_array) - weighting of each choice
                [N X 1]

    Returns:
    i         - (int)      - chosen index
    """
    minVal = np.min(weights)
    weights = weights - minVal # handle negative vals
    cumVal = np.cumsum(weights)
    pick = np.random.uniform(0, cumVal[-1])
    for i in range(len(weights)):
        if cumVal[i] >= pick:
            return i


def selectAct(action, actSelect):  
    """Selects action based on vector of actions

    Single Action:
    - Hard: a single action is chosen based on the highest index
    - Prob: a single action is chosen probablistically with higher values
            more likely to be chosen

    We aren't selecting a single action:
    - Softmax: a softmax normalized distribution of values is returned
    - Default: all actions are returned 

    Args:
    action   - (np_array) - vector weighting each possible action
                [N X 1]

    Returns:
    i         - (int) or (np_array)     - chosen index
                         [N X 1]
    """
    if actSelect == 'softmax':
        action = softmax(action)
    elif actSelect == 'prob':
        action = weightedRandom(np.sum(action,axis=0))
    else:
        action = action.flatten()
    return action


def act(weights, aVec, nInput, nOutput, inPattern):
    """Returns FFANN output given a single input pattern
    If the variable weights is a vector it is turned into a square weight matrix.

    Allows the network to return the result of several samples at once if given a matrix instead of a vector of inputs:
      Dim 0 : individual samples
      Dim 1 : dimensionality of pattern (# of inputs)

    Args:
    weights   - (np_array) - ordered weight matrix or vector
                [N X N] or [N**2]
    aVec      - (np_array) - activation function of each node
                [N X 1]    - stored as ints (see applyAct in ann.py)
    nInput    - (int)      - number of input nodes
    nOutput   - (int)      - number of output nodes
    inPattern - (np_array) - input activation
                [1 X nInput] or [nSamples X nInput]

    Returns:
    output    - (np_array) - output activation
                [1 X nOutput] or [nSamples X nOutput]
    """
    # Turn weight vector into weight matrix
    if np.ndim(weights) < 2:
        nNodes = int(np.sqrt(np.shape(weights)[0]))
        wMat = np.reshape(weights, (nNodes, nNodes))
    else:
        nNodes = np.shape(weights)[0]
        wMat = weights
    wMat[np.isnan(wMat)]=0

    # Vectorize input
    if np.ndim(inPattern) > 1:
        nSamples = np.shape(inPattern)[0]
    else:
        nSamples = 1

    # Run input pattern through ANN    
    nodeAct  = np.zeros((nSamples,nNodes))
    nodeAct[:,0] = 1 # Bias activation
    nodeAct[:,1:nInput+1] = inPattern

    # Propagate signal through hidden to output nodes
    iNode = nInput+1
    for iNode in range(nInput+1,nNodes):
        rawAct = np.dot(nodeAct, wMat[:,iNode]).squeeze()
        nodeAct[:,iNode] = applyAct(aVec[iNode], rawAct) 
        #print(nodeAct)
    output = nodeAct[:,-nOutput:]   
    return output


def applyAct(actId, x):
    """Returns value after an activation function is applied
    Lookup table to allow activations to be stored in numpy arrays

    case 1  -- Linear
    case 2  -- Unsigned Step Function
    case 3  -- Sin
    case 4  -- Gausian with mean 0 and sigma 1
    case 5  -- Hyperbolic Tangent [tanh] (signed)
    case 6  -- Sigmoid unsigned [1 / (1 + exp(-x))]
    case 7  -- Inverse
    case 8  -- Absolute Value
    case 9  -- Relu
    case 10 -- Cosine
    case 11 -- Squared

    Args:
    actId   - (int)   - key to look up table
    x       - (???)   - value to be input into activation
              [? X ?] - any type or dimensionality

    Returns:
    output  - (float) - value after activation is applied
              [? X ?] - same dimensionality as input
    """
    if actId == 1:   # Linear
        value = x

    if actId == 2:   # Unsigned Step Function
        value = 1.0*(x>0.0)
    #value = (np.tanh(50*x/2.0) + 1.0)/2.0

    elif actId == 3: # Sin
        value = np.sin(np.pi*x) 

    elif actId == 4: # Gaussian with mean 0 and sigma 1
        value = np.exp(-np.multiply(x, x) / 2.0)

    elif actId == 5: # Hyperbolic Tangent (signed)
        value = np.tanh(x)     

    elif actId == 6: # Sigmoid (unsigned)
        value = (np.tanh(x/2.0) + 1.0)/2.0

    elif actId == 7: # Inverse
        value = -x

    elif actId == 8: # Absolute Value
        value = abs(x)   

    elif actId == 9: # Relu
        value = np.maximum(0, x)   

    elif actId == 10: # Cosine
        value = np.cos(np.pi*x)

    elif actId == 11: # Squared
        value = x**2

    else:
        value = x

    return value

# End of copypaste
# -------------------------------------------------------------------


# This action is original to this repository
def create_wann_agent(agent_path, agent_type, env):
    """
    Load and return a WANN agent.
    The agent has a function `get_action` that takes in
    an observation and returns an appropiate action.
    """
    np_data = np.load(agent_path)
    wMat = np_data["wMat"]
    aVec = np_data["aVec"]

    # TODO support for other input spaces?
    nInput = env.observation_space.shape[0]
    nOutput = 0
    action_type = "all"
    if isinstance(env.action_space, spaces.Box):
        nOutput = env.action_space.shape[0]
    elif isinstance(env.action_space, spaces.Discrete):
        nOutput = env.action_space.n
        action_type = "prob"
    else:
        raise ValueError("Unsupported action space")

    def get_action(obs):
        # Includes batch-size
        output = act(wMat, aVec, nInput, nOutput, obs)
        action = selectAct(output, action_type)
        return action

    agent = SimpleAgentClass(lambda obs: get_action(obs))
    return agent
