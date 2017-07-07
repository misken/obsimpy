import simpy
import numpy as np
from numpy.random import RandomState

"""
Simple OB patient flow model - NOT OO

Details:

- Generate arrivals via Poisson process
- Simple delays modeling stay in units. No resource contention modeled.

Scenario:
  Patients arrive according to Poisson process and then proceed through
  delays representing stays at OBS, LDR, and PP units.

"""

MEAN_LOS_OBS = 3
MEAN_LOS_LDR = 12
MEAN_LOS_PP = 48

RNG_SEED = 6353

def source(env, number, arr_rate, prng):
    """Source generates patients according to a simple Poisson process"""
    for i in range(number):

        iat = prng.exponential(1.0 / arr_rate)

        # Hard coding for now

        los_obs = prng.exponential(MEAN_LOS_OBS)
        los_ldr = prng.exponential(MEAN_LOS_LDR)
        los_pp = prng.exponential(MEAN_LOS_PP)

        obp = obpatient_flow(env, 'Patient{}'.format(i),
                             los_obs=los_obs, los_ldr=los_ldr, los_pp=los_pp)

        env.process(obp)

        yield env.timeout(iat)

# Define an obpatient_flow "process"
def obpatient_flow(env, name, los_obs, los_ldr, los_pp):
    while True:
        print("{} entering OBS at {}".format(name, env.now))
        yield env.timeout(los_obs)

        print("{} entering LDR at {}".format(name, env.now))
        yield env.timeout(los_ldr)

        print("{} entering PP at {}".format(name, env.now))
        yield env.timeout(los_pp)


# Initialize a simulation environment
env = simpy.Environment()

prng = RandomState(RNG_SEED)

# Create a process generator and start it and add it to the env
# Calling obpatient(env) creates the generator.
# env.process() starts and adds it to env
env.process(source(env, 1000, 0.2, prng))

# Run the simulation for a while
runtime = 250
env.run(until=runtime)
