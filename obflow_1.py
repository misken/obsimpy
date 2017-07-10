import simpy
import numpy as np
from numpy.random import RandomState

"""
Simple OB patient flow model - NOT OO

Details:

- Generate arrivals via Poisson process
- Simple delays modeling stay in OBS, LDR, and PP units. No capacitated resource contention modeled.


"""

ARR_RATE = 0.4
MEAN_LOS_OBS = 3
MEAN_LOS_LDR = 12
MEAN_LOS_PP = 48

RNG_SEED = 6353

def source(env, arr_rate, stoptime=250, prng=0):
    """Source generates patients according to a simple Poisson process"""

    patients_created = 0
    while env.now < stoptime:

        iat = prng.exponential(1.0 / arr_rate)

        # Hard coding for now

        los_obs = prng.exponential(MEAN_LOS_OBS)
        los_ldr = prng.exponential(MEAN_LOS_LDR)
        los_pp = prng.exponential(MEAN_LOS_PP)

        patients_created += 1
        obp = obpatient_flow(env, 'Patient{}'.format(patients_created),
                             los_obs=los_obs, los_ldr=los_ldr, los_pp=los_pp)

        env.process(obp)

        yield env.timeout(iat)

# Define an obpatient_flow "process"
def obpatient_flow(env, name, los_obs, los_ldr, los_pp):

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
runtime = 250
env.process(source(env, ARR_RATE, runtime, prng))
env.run()