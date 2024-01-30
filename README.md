# A Minimal Cortex-Basal Ganglia Architecture
## Features
* [BriCA](https://github.com/wbap/BriCA1)
* The exterior environment with OpenAI Gym
* Action predictor (Cortex) -- a perceptron
* Go/NoGo determiner (BG) with reinforcement/frequency learning

For details, please read [this article](https://rondelionai.blogspot.com/2021/12/a-minimal-cortex-basal-ganglia.html) (based on **CBT1cCA_1.py**).

**What's New:**
* PyTorch DQN version: **CBT1cCA_3.py**
* Generic TensorForce RL version: **CBT1cCA_2.py**

## How to Install
* Clone the repository
* BriCA1
    * Follow the instruction [here](http://wbap.github.io/BriCA1/tutorial/introduction.html#installing).
* pip install the following
    * Numpy
    * [OpenAI Gym](https://gym.openai.com/)
    * [PyTorch](https://pytorch.org)
    * [TensorForce](https://github.com/tensorforce/tensorforce) (not necessary for CBT1cCA_3.py)
* Register the environment to Gym
    * Place `CBT1Env.py` file in `gym/gym/envs/myenv`  
    (wherever Gym to be used is installed)
    * Add to `__init__.py` (located in the same folder)  
      `from gym.envs.myenv.CBT1Env import CBT1Env`
    * Add to `gym/gym/envs/__init__.py`  
```
register(
    id='CBT1Env-v0',
    entry_point='gym.envs.myenv:CBT1Env'
    )
```

## Usage
### Command arguments
- First arg: 1:random act, 2: reinforcement learning, 3: frequency learning  
(CBT1cCA_2/CBT1cCA_3 do not have the option 3.)  

- Options
      --dump: dump file path')
      --episode_count: Number of training episodes (default: 1)
      --max_steps: Max steps in an episode (default: 20)
      --config: Model configuration (default: CBT1CA.json)
      --dump_flags: m:main, b:bg, o:obs, p:predictor

### Sample usage
```
$ python CBT1cCA_1.py 3 --episode_count 4000 --dump "BG_dump.txt" --dump_flags "mbp"

```

## Other files

* CBT1Env.py:  Test env. (note: not MDP for delay > 1)
* CBT1CA.json:	config. file for CBT1CA_1
* CBT1CA2.json:	config. file for CBT1CA_2  
Use always "dqn" for the RL agent.
* CBT1CA3.json: config. file for CBT1CA_3
* CBT1EnvRLTest.py:	an RL agent for the environment
* CBT1CA.brical.json: BriCA Language file for CBT1cCA_1_BL
* dqn.py: PyTorch DQN module (based on a PyTorch tutorial)


