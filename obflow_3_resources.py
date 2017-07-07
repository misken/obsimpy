import simpy
import numpy as np
from numpy.random import RandomState

"""
Simple OB patient flow model 3 - NOT OO

Details:

- Generate arrivals via Poisson process
- Uses Resource objects to model OBS, LDR, and PP

"""
ARR_RATE = 0.4
MEAN_LOS_OBS = 3
MEAN_LOS_LDR = 12
MEAN_LOS_PP = 48

CAPACITY_OBS = 2
CAPACITY_LDR = 6
CAPACITY_PP = 24

RNG_SEED = 6353

def patient_generator(env, arr_stream, arr_rate, initial_delay=0, stoptime=250, prng=0):
    """Generate patients according to a simple Poisson process"""

    patients_created = 0
    yield env.timeout(initial_delay)
    while env.now < stoptime:

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

        # Compute next interarrival time

        yield env.timeout(iat)

# Define an obpatient_flow "process"
def obpatient_flow(env, name, los_obs, los_ldr, los_pp):

    # OBS
    print("{} trying to get OBS at {}".format(name, env.now))
    bed_request_ts = env.now
    bed_request1 = obs_unit.request() # Request an obs bed
    yield bed_request1
    print("{} entering OBS at {}".format(name, env.now))
    if env.now > bed_request_ts:
        print("{} waited {} time units for OBS bed".format(name, env.now-  bed_request_ts))
    yield env.timeout(los_obs) # Stay in obs bed

    print("{} trying to get LDR at {}".format(name, env.now))
    bed_request_ts = env.now
    bed_request2 = ldr_unit.request()  # Request an obs bed
    yield bed_request2

    # Got LDR bed, release OBS bed
    obs_unit.release(bed_request1)  # Release the obs bed
    print("{} leaving OBS at {}".format(name, env.now))

    # LDR stay
    print("{} entering LDR at {}".format(name, env.now))
    if env.now > bed_request_ts:
        print("{} waited {} time units for LDR bed".format(name, env.now-  bed_request_ts))
    yield env.timeout(los_ldr) # Stay in LDR bed

    print("{} trying to get PP at {}".format(name, env.now))
    bed_request_ts = env.now
    bed_request3 = pp_unit.request()  # Request a PP bed
    yield bed_request3

    # Got PP bed, release LDR bed
    ldr_unit.release(bed_request2)  # Release the obs bed
    print("{} leaving LDR at {}".format(name, env.now))

    # PP stay
    print("{} entering PP at {}".format(name, env.now))
    if env.now > bed_request_ts:
        print("{} waited {} time units for PP bed".format(name, env.now-  bed_request_ts))
    yield env.timeout(los_pp) # Stay in LDR bed
    pp_unit.release(bed_request3)  # Release the PP bed

    print("{} leaving PP and system at {}".format(name, env.now))

# Initialize a simulation environment
env = simpy.Environment()

prng = RandomState(RNG_SEED)

rho_obs = ARR_RATE * MEAN_LOS_OBS / CAPACITY_OBS
rho_ldr = ARR_RATE * MEAN_LOS_LDR / CAPACITY_LDR
rho_pp = ARR_RATE * MEAN_LOS_PP / CAPACITY_PP

print(rho_obs, rho_ldr, rho_pp)

# Declare a Resource to model OBS unit
obs_unit = simpy.Resource(env, CAPACITY_OBS)
ldr_unit = simpy.Resource(env, CAPACITY_LDR)
pp_unit = simpy.Resource(env, CAPACITY_PP)

# Run the simulation for a while
runtime = 250
env.process(patient_generator(env, "Type1", ARR_RATE, 0, runtime, prng))
env.run()