# A Minimal Cortex-Basal Ganglia Architecture
## Features
* [BriCA](https://github.com/wbap/BriCA1)
* The exterior environment with OpenAI Gym
* Action predictor (Cortex) -- a perceptron
* Go/NoGo determiner (BG) with reinforcement/frequency learning

For details, please read [this article](https://rondelionai.blogspot.com/2021/12/a-minimal-cortex-basal-ganglia.html).

## How to Install
* Clone the repository
* BriCA1
    * Follow the instruction [here](http://wbap.github.io/BriCA1/tutorial/introduction.html#installing).
* pip install the following
    * Numpy
    * [OpenAI Gym](https://gym.openai.com/)
    * [PyTorch](https://pytorch.org)
    * [TensorForce](https://github.com/tensorforce/tensorforce)



## Usage
### Command arguments
- First arg: 1:random act, 2: reinforcement learning, 3: frequency learning
- Options
      --dump: dump file path')
      --episode_count: Number of training episodes (default: 1)
      --max_steps: Max steps in an episode (default: 20)
      --config: Model configuration (default: CBT1CA.json)
      --dump_flags: m:main, b:bg, o:obs, p:predictor

### Sample usage
```
$ python CBT1cCA.py 3 --episode_count 4000 --dump "BG_dump.txt" --dump_flags "mbp"

```

## Other files

* CBT1CA.json:	config. file
* CBT1Env.py:	test environment
* CBT1EnvRLTest.py:	an RL agent for the environment


