import simpy
import numpy as np
from numpy.random import RandomState

"""
Simple OB patient flow model 2 - NOT OO

Details:

- Generate arrivals via Poisson process
- Uses one Resource objects to model OBS, the other units are modeled as simple delays.
- Arrival rates and mean lengths of stay hard coded as constants. Later versions will read these from input files.

"""
# Arrival rate and length of stay inputs.
ARR_RATE = 0.4
MEAN_LOS_OBS = 3
MEAN_LOS_LDR = 12
MEAN_LOS_PP = 48

CAPACITY_OBS = 2

RNG_SEED = 6353

def patient_generator(env, arr_rate, prng=RandomState(0)):
    """Generates patients according to a simple Poisson process

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        arr_rate : float
            exponential arrival rate
        prng : RandomState object
            Seeded RandomState object for generating pseudo-random numbers.
            See https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html

    """

    patients_created = 0

    # Infinite loop for generatirng patients according to a poisson process.
    while True:

        # Generate next interarrival time
        iat = prng.exponential(1.0 / arr_rate)

        # Generate length of stay in each unit for this patient
        los_obs = prng.exponential(MEAN_LOS_OBS)
        los_ldr = prng.exponential(MEAN_LOS_LDR)
        los_pp = prng.exponential(MEAN_LOS_PP)

        # Update counter of patients
        patients_created += 1

        # Create a new patient flow process.
        obp = obpatient_flow(env, 'Patient{}'.format(patients_created),
                             los_obs=los_obs, los_ldr=los_ldr, los_pp=los_pp)

        # Register the process with the simulation environment
        env.process(obp)

        # This process will now yield to a 'timeout' event. This process will resume after iat time units.
        yield env.timeout(iat)


def obpatient_flow(env, name, los_obs, los_ldr, los_pp):
    """Process function modeling how a patient flows through system.

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        name : str
            process instance id
        los_obs : float
            length of stay in OBS unit
        los_ldr : float
            length of stay in LDR unit
        los_pp : float
            length of stay in PP unit
    """

    print("{} trying to get OBS at {}".format(name, env.now))

    # Timestamp when patient tried to get OBS bed
    bed_request_ts = env.now
    # Request an obs bed
    bed_request = obs_unit.request()
    # Yield this process until a bed is available
    yield bed_request

    # We got an OBS bed
    print("{} entering OBS at {}".format(name, env.now))
    # Let's see if we had to wait to get the bed.
    if env.now > bed_request_ts:
        print("{} waited {} time units for OBS bed".format(name, env.now - bed_request_ts))

    # Yield this process again. Now wait until our length of stay elapses.
    # This is the actual stay in the bed
    yield env.timeout(los_obs)

    # All done with OBS, release the bed. Note that we pass the bed_request object
    # to the release() function so that the correct unit of the resource is released.
    obs_unit.release(bed_request)
    print("{} leaving OBS at {}".format(name, env.now))

    # Continue on through LDR and PP; modeled as simple delays for now.

    print("{} entering LDR at {}".format(name, env.now))
    yield env.timeout(los_ldr)

    print("{} entering PP at {}".format(name, env.now))
    yield env.timeout(los_pp)


# Initialize a simulation environment
env = simpy.Environment()

# Initialize a random number generator.
# See https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html
prng = RandomState(RNG_SEED)

# Declare a Resource to model OBS unit. Default capacity is 1, we pass in desired capacity.
obs_unit = simpy.Resource(env, CAPACITY_OBS)

# Run the simulation for a while
runtime = 250
env.process(patient_generator(env, ARR_RATE, prng))
env.run(until=runtime)
