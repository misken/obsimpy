import simpy
import numpy as np
from numpy.random import RandomState

"""
Simple OB patient flow model 3 - NOT OO

Details:

- Generate arrivals via Poisson process
- Uses one Resource objects to model OBS, LDR, and PP.
- Arrival rates and mean lengths of stay hard coded as constants. Later versions will read these from input files.
- Additional functionality added to arrival generator (initial delay and arrival stop time).

"""
ARR_RATE = 0.4
MEAN_LOS_OBS = 3
MEAN_LOS_LDR = 12
MEAN_LOS_PP = 48

CAPACITY_OBS = 2
CAPACITY_LDR = 6
CAPACITY_PP = 24

RNG_SEED = 6353

def patient_generator(env, arr_stream, arr_rate, initial_delay=0,
                      stoptime=simpy.core.Infinity, prng=RandomState(0)):
    """Generates patients according to a simple Poisson process

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        arr_rate : float
            exponential arrival rate
        initial_delay: float (default 0)
            time before arrival generation should begin
        stoptime: float (default Infinity)
            time after which no arrivals are generated
        prng : RandomState object
            Seeded RandomState object for generating pseudo-random numbers.
            See https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html

    """

    patients_created = 0

    # Yield for the initial delay
    yield env.timeout(initial_delay)

    # Generate arrivals as long as simulation time is before stoptime
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

    # Note the repetitive code and the use of separate request objects for each
    # stay in the different units.

    # OBS
    print("{} trying to get OBS at {}".format(name, env.now))
    bed_request_ts = env.now
    bed_request1 = obs_unit.request() # Request an OBS bed
    yield bed_request1
    print("{} entering OBS at {}".format(name, env.now))
    if env.now > bed_request_ts:
        print("{} waited {} time units for OBS bed".format(name, env.now-  bed_request_ts))
    yield env.timeout(los_obs) # Stay in obs bed

    print("{} trying to get LDR at {}".format(name, env.now))
    bed_request_ts = env.now
    bed_request2 = ldr_unit.request()  # Request an LDR bed
    yield bed_request2

    # Got LDR bed, release OBS bed
    obs_unit.release(bed_request1)  # Release the OBS bed
    print("{} leaving OBS at {}".format(name, env.now))

    # LDR stay
    print("{} entering LDR at {}".format(name, env.now))
    if env.now > bed_request_ts:
        print("{} waited {} time units for LDR bed".format(name, env.now - bed_request_ts))
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
        print("{} waited {} time units for PP bed".format(name, env.now - bed_request_ts))
    yield env.timeout(los_pp) # Stay in PP bed
    pp_unit.release(bed_request3)  # Release the PP bed

    print("{} leaving PP and system at {}".format(name, env.now))

# Initialize a simulation environment
env = simpy.Environment()

prng = RandomState(RNG_SEED)

rho_obs = ARR_RATE * MEAN_LOS_OBS / CAPACITY_OBS
rho_ldr = ARR_RATE * MEAN_LOS_LDR / CAPACITY_LDR
rho_pp = ARR_RATE * MEAN_LOS_PP / CAPACITY_PP

print(rho_obs, rho_ldr, rho_pp)

# Declare Resources to model all units
obs_unit = simpy.Resource(env, CAPACITY_OBS)
ldr_unit = simpy.Resource(env, CAPACITY_LDR)
pp_unit = simpy.Resource(env, CAPACITY_PP)

# Run the simulation for a while. Let's shut arrivals off after 100 time units.
runtime = 250
stop_arrivals = 100
env.process(patient_generator(env, "Type1", ARR_RATE, 0, stop_arrivals, prng))
env.run(until=runtime)