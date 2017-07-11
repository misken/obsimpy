import simpy
import numpy as np
from numpy.random import RandomState

"""
Simple OB patient flow model - NOT OO

Details:

- Generate arrivals via Poisson process
- Simple delays modeling stay in OBS, LDR, and PP units. No capacitated resource contention modeled.
- Arrival rates and mean lengths of stay hard coded as constants. Later versions will read these from input files.

"""

# Arrival rate and length of stay inputs.
ARR_RATE = 0.4
MEAN_LOS_OBS = 3
MEAN_LOS_LDR = 12
MEAN_LOS_PP = 48

RNG_SEED = 6353

def source(env, arr_rate, prng=RandomState(0)):
    """Source generates patients according to a simple Poisson process

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

# Define an obpatient_flow "process"
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

    # Note how we are simply modeling each stay as a delay. There
    # is NO contention for any resources.
    print("{} entering OBS at {}".format(name, env.now))
    yield env.timeout(los_obs)

    print("{} entering LDR at {}".format(name, env.now))
    yield env.timeout(los_ldr)

    print("{} entering PP at {}".format(name, env.now))
    yield env.timeout(los_pp)


# Initialize a simulation environment
env = simpy.Environment()

# Initialize a random number generator.
# See https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html
prng = RandomState(RNG_SEED)

# Create a process generator and start it and add it to the env
# Calling obpatient(env) creates the generator.
# env.process() starts and adds it to env
runtime = 250
env.process(source(env, ARR_RATE, prng))

# Run the simulation
env.run(until=runtime)