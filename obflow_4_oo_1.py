import simpy
import numpy as np
from numpy.random import RandomState



"""
Simple OB patient flow model 4 - Very simple OO

Details:

- Generate arrivals via Poisson process
- Define an OBUnit class that contains a simpy.Resource object as a member.
  Not subclassing Resource, just trying to use it as a member.
- Routing is hard coded (no Router class yet)
- Just trying to get objects/processes to communicate

"""

ARR_RATE = 0.4
MEAN_LOS_OBS = 3
MEAN_LOS_LDR = 12
MEAN_LOS_PP = 48

CAPACITY_OBS = 2
CAPACITY_LDR = 6
CAPACITY_PP = 24

RNG_SEED = 6353


class OBunit(object):
    """ Models an OB unit with fixed capacity.

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        name : str
            unit name
        capacity : integer (or None)
            Number of beds. Use None for infinite capacity.

    """

    def __init__(self, env, name, capacity=None, debug=False):
        if capacity is None:
            self.capacity = capacity=simpy.core.Infinity
        else:
            self.capacity = capacity

        self.unit = simpy.Resource(env, capacity)
        self.env = env
        self.name = name

        self.debug = debug
        self.num_entries = 0
        self.num_exits = 0
        self.tot_occ_time = 0.0

    def basic_stats_msg(self):

        msg = "{:6}:\t Entries={}, Exits={}, ALOS={:4.2f}".format(self.name,
                                                              self.num_entries,
                                                              self.num_exits,
                                                              self.tot_occ_time / self.num_exits)
        return msg


class OBpatient(object):
    def __init__(self, arrtime, arrstream, patient_id=0, prng=0):
        self.arrtime = arrtime
        self.arrstream = arrstream
        self.patient_id = patient_id
        self.name = 'Patient_{}'.format(patient_id)

        # Hard coding for now

        self.planned_los_obs = prng.exponential(MEAN_LOS_OBS)
        self.planned_los_ldr = prng.exponential(MEAN_LOS_LDR)
        self.planned_los_pp = prng.exponential(MEAN_LOS_PP)

    def __repr__(self):
        return "patientid: {}, arr_stream: {}, time: {}". \
            format(self.patient_id, self.arrstream, self.arrtime)


class OBPatientGenerator(object):
    """ Generates patients.

        Set the "out" member variable to the resource at which patient generated.

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        adist : function
            a no parameter function that returns the successive inter-arrival times of the packets
        initial_delay : number
            Starts generation after an initial delay. Default = 0
        stoptime : number
            Stops generation at the stoptime. Default is infinite

    """

    def __init__(self, env, arr_stream, arr_rate, initial_delay=0, stoptime=250, debug=False):
        self.id = id
        self.env = env
        self.arr_rate = arr_rate
        self.arr_stream = arr_stream
        self.initial_delay = initial_delay
        self.stoptime = stoptime
        self.debug = debug
        self.num_patients_created = 0

        self.prng = RandomState(RNG_SEED)

        self.action = env.process(self.run())  # starts the run() method as a SimPy process


    def run(self):
        """The patient generator.
        """
        # Delay for initial_delay
        yield self.env.timeout(self.initial_delay)
        # Main generator loop that terminates when stoptime reached
        while self.env.now < self.stoptime:
            # Delay until time for next arrival
            # Compute next interarrival time
            iat = self.prng.exponential(1.0 / self.arr_rate)
            yield self.env.timeout(iat)
            self.num_patients_created += 1
            # Create new patient
            obp = OBpatient(self.env.now, self.arr_stream, patient_id=self.num_patients_created, prng=self.prng)
            # Create a new flow instance for this patient. The OBpatient object carries all necessary info.
            obflow = obpatient_flow(env, obp, self.debug)
            # Register the new flow instance as a SimPy process.
            self.env.process(obflow)


def obpatient_flow(env, obp, debug=False):
    """ Models the patient flow process.

        The sequence of units is hard coded for now.

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        obp : OBpatient object
            the patient to send through the flow process

    """

    name = obp.name

    # OBS
    if debug:
        print("{} trying to get OBS at {}".format(name, env.now))
    bed_request_ts = env.now
    bed_request1 = obs_unit.unit.request() # Request an obs bed
    yield bed_request1
    if debug:
        print("{} entering OBS at {}".format(name, env.now))
    obs_unit.num_entries += 1
    enter_ts = env.now
    if debug:
        if env.now > bed_request_ts:
            print("{} waited {} time units for OBS bed".format(name, env.now-  bed_request_ts))

    yield env.timeout(obp.planned_los_obs) # Stay in obs bed

    if debug:
        print("{} trying to get LDR at {}".format(name, env.now))
    bed_request_ts = env.now
    bed_request2 = ldr_unit.unit.request()  # Request an obs bed
    yield bed_request2

    # Got LDR bed, release OBS bed
    obs_unit.unit.release(bed_request1)  # Release the obs bed
    obs_unit.num_exits += 1
    exit_ts = env.now
    obs_unit.tot_occ_time += exit_ts - enter_ts
    if debug:
        print("{} leaving OBS at {}".format(name, env.now))

    # LDR stay
    if debug:
        print("{} entering LDR at {}".format(name, env.now))
    ldr_unit.num_entries += 1
    enter_ts = env.now
    if debug:
        if env.now > bed_request_ts:
            print("{} waited {} time units for LDR bed".format(name, env.now-  bed_request_ts))
    yield env.timeout(obp.planned_los_ldr) # Stay in LDR bed

    if debug:
        print("{} trying to get PP at {}".format(name, env.now))
    bed_request_ts = env.now
    bed_request3 = pp_unit.unit.request()  # Request a PP bed
    yield bed_request3

    # Got PP bed, release LDR bed
    ldr_unit.unit.release(bed_request2)  # Release the ldr bed
    ldr_unit.num_exits += 1
    exit_ts = env.now
    ldr_unit.tot_occ_time += exit_ts - enter_ts
    if debug:
        print("{} leaving LDR at {}".format(name, env.now))

    # PP stay
    if debug:
        print("{} entering PP at {}".format(name, env.now))
    pp_unit.num_entries += 1
    enter_ts = env.now
    if debug:
        if env.now > bed_request_ts:
            print("{} waited {} time units for PP bed".format(name, env.now-  bed_request_ts))
    yield env.timeout(obp.planned_los_pp) # Stay in LDR bed
    pp_unit.unit.release(bed_request3)  # Release the PP bed
    pp_unit.num_exits += 1
    exit_ts = env.now
    pp_unit.tot_occ_time += exit_ts - enter_ts

    if debug:
        print("{} leaving PP and system at {}".format(name, env.now))



# Initialize a simulation environment
env = simpy.Environment()

rho_obs = ARR_RATE * MEAN_LOS_OBS / CAPACITY_OBS
rho_ldr = ARR_RATE * MEAN_LOS_LDR / CAPACITY_LDR
rho_pp = ARR_RATE * MEAN_LOS_PP / CAPACITY_PP

print("rho_obs: {:6.3f}, rho_ldr: {:6.3f}, rho_pp: {:6.3f}".format(rho_obs, rho_ldr, rho_pp))


# Declare a Resource to model OBS unit
obs_unit = OBunit(env, "OBS", CAPACITY_OBS, debug=True)
ldr_unit = OBunit(env, "LDR", CAPACITY_LDR, debug=True)
pp_unit = OBunit(env, "PP", CAPACITY_PP, debug=True)

# Run the simulation for a while
runtime = 100000
debug = False

obpat_gen = OBPatientGenerator(env, "Type1", ARR_RATE, 0, runtime, debug=debug)
env.run()

print("\nNum patients generated: {}\n".format(obpat_gen.num_patients_created))
print(obs_unit.basic_stats_msg())
print(ldr_unit.basic_stats_msg())
print(pp_unit.basic_stats_msg())


