import simpy
import numpy as np
from numpy.random import RandomState

"""
Simple OB patient flow model 2 - NOT OO

Details:

- Generate arrivals via Poisson process
- Uses one Resource objects to model OBS

Scenario:
  Patients arrive according to Poisson process. The OBS unit is modeled as a Resource
  and after leaving OBS, the other units are modeled as simple delays.

"""
ARR_RATE = 0.2
MEAN_LOS_OBS = 3
MEAN_LOS_LDR = 12
MEAN_LOS_PP = 48

CAPACITY_OBS = 2

RNG_SEED = 6353

def patient_generator(env, arr_rate, stoptime=250, prng=0):
    """Generate patients according to a simple Poisson process"""
    patients_created = 0
    while env.now < stoptime:

        # Compute next interarrival time
        iat = prng.exponential(1.0 / arr_rate)

        # Sample los distributions
        los_obs = prng.exponential(MEAN_LOS_OBS)
        los_ldr = prng.exponential(MEAN_LOS_LDR)
        los_pp = prng.exponential(MEAN_LOS_PP)


        # Create new patient process instance
        patients_created += 1
        obp = obpatient_flow(env, 'Patient{}'.format(patients_created),
                             los_obs=los_obs, los_ldr=los_ldr, los_pp=los_pp)

        env.process(obp)


        yield env.timeout(iat)

# Define an obpatient_flow "process"
def obpatient_flow(env, name, los_obs, los_ldr, los_pp):

    print("{} trying to get OBS at {}".format(name, env.now))
    bed_request_ts = env.now
    bed_request = obs_unit.request() # Request an obs bed
    yield bed_request
    print("{} entering OBS at {}".format(name, env.now))
    if env.now > bed_request_ts:
        print("{} waited {} time units for OBS bed".format(name, env.now-  bed_request_ts))


    yield env.timeout(los_obs) # Stay in obs bed
    obs_unit.release(bed_request)  # Release the obs bed

    print("{} leaving OBS at {}".format(name, env.now))

    print("{} entering LDR at {}".format(name, env.now))
    yield env.timeout(los_ldr)

    print("{} entering PP at {}".format(name, env.now))
    yield env.timeout(los_pp)


# Initialize a simulation environment
env = simpy.Environment()

prng = RandomState(RNG_SEED)

# Declare a Resource to model OBS unit
obs_unit = simpy.Resource(env, CAPACITY_OBS)

# Run the simulation for a while
runtime = 250
env.process(patient_generator(env, ARR_RATE, runtime, prng))
env.run()
