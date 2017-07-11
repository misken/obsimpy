import simpy
import numpy as np
from numpy.random import RandomState


"""
Simple OB patient flow model 5 - Very simple OO

Details:

- Generate arrivals via Poisson process
- Define an OBUnit class that contains a simpy.Resource object as a member.
  Not subclassing Resource, just trying to use it as a member.
- Routing is done via setting ``out`` member of an OBUnit instance to
 another OBUnit instance to which the OB patient flow instance should be
 routed. The routing logic, for now, is in the main script. In addition,
 the sequence of lengths of stay
 Later,
 we need some sort of router object and data driven routing.
- Trying to get patient flow working without a process function that
explicitly articulates the sequence of units and stays.

Key Lessons Learned:

- Any function that is a generator and might potentially yield for an event must get registered
  as a process. For

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

        self.out = None


    def put(self, obp):

        if self.debug:
            print("{} trying to get {} at {}".format(obp.name, self.name, env.now))

        obp.current_stay_num += 1
        # Timestamp of request time
        bed_request_ts = env.now
        # Request a bed
        bed_request = self.unit.request()
        obp.bed_requests[obp.current_stay_num] = bed_request
        obp.request_entry_ts[obp.current_stay_num] = env.now
        yield bed_request

        # Seized a bed.
        obp.entry_ts[obp.current_stay_num] = env.now

        # Check if we have a bed from a previous stay and release it. Update stats for
        # previous unit.

        if obp.bed_requests[obp.current_stay_num - 1] is not None:
            previous_request = obp.bed_requests[obp.current_stay_num - 1]
            previous_unit_name = obp.planned_route_stop[obp.current_stay_num - 1]
            previous_unit = obunits[previous_unit_name]
            previous_unit.unit.release(previous_request)
            previous_unit.num_exits += 1
            previous_unit.tot_occ_time += env.now - obp.entry_ts[obp.current_stay_num - 1]
            obp.exit_ts[obp.current_stay_num - 1] = env.now

        if self.debug:
            print("{} entering {} at {}".format(obp.name, self.name, env.now))
        self.num_entries += 1
        enter_ts = env.now
        if self.debug:
            if env.now > bed_request_ts:
                print("{} waited {} time units for {} bed".format(obp.name, env.now - bed_request_ts, self.name))

        # Determine los and then yield for the stay
        los = obp.planned_los[obp.current_stay_num]
        yield env.timeout(los)

        # Go to next destination (which could be an exitflow)
        next_unit_name = obp.planned_route_stop[obp.current_stay_num + 1]
        self.out = obunits[next_unit_name]
        if obp.current_stay_num == obp.route_length:
            self.out.put(obp)
        else:
            self.env.process(self.out.put(obp))

    def get(self, unit):
        return unit.get()

    def basic_stats_msg(self):

        if self.num_exits > 0:
            alos = self.tot_occ_time / self.num_exits
        else:
            alos = 0

        msg = "{:6}:\t Entries={}, Exits={}, ALOS={:4.2f}".format(self.name,
                                                              self.num_entries,
                                                              self.num_exits,
                                                              alos)
        return msg

class ExitFlow(object):
    """ Patients routed here when ready to exit.

        Patient objects put into a Store. Can be accessed later for stats and logs. A little
        worried about how big the Store will get.

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        debug : boolean
            if true then patient details printed on arrival


    """

    def __init__(self, env, name, store_obp=True, debug=False):
        self.store = simpy.Store(env)
        self.env = env
        self.name = name
        self.store_obp = store_obp
        self.debug = debug
        self.num_exits = 0
        self.last_exit = 0.0

    def put(self, obp):

        if obp.bed_requests[obp.current_stay_num] is not None:
            previous_request = obp.bed_requests[obp.current_stay_num]
            previous_unit_name = obp.planned_route_stop[obp.current_stay_num]
            previous_unit = obunits[previous_unit_name]
            previous_unit.unit.release(previous_request)
            previous_unit.num_exits += 1
            previous_unit.tot_occ_time += env.now - obp.entry_ts[obp.current_stay_num]
            obp.exit_ts[obp.current_stay_num - 1] = env.now

        self.last_exit = self.env.now
        self.num_exits += 1

        if self.debug:
            print(obp)

        # Store patient
        if self.store_obp:
            self.store.put(obp)


    def basic_stats_msg(self):
        msg = "{:6}:\t Exits={}, Last Exit={:10.2f}".format(self.name,
                                                                  self.num_exits,
                                                                  self.last_exit)


class OBPatient(object):
    def __init__(self, arrtime, arrstream, obunits, patient_id=0, prng=None):
        self.arrtime = arrtime
        self.arrstream = arrstream
        self.patient_id = patient_id
        self.name = 'Patient_{}'.format(patient_id)

        # Hard coding route, los and bed requests for now
        # Not sure how best to do routing related data structures.
        # Hack for now using combination of lists here, the out member
        # and the obunits dictionary.
        self.current_stay_num = 0
        self.route_length = 3

        self.planned_route_stop = []
        self.planned_route_stop.append(None)
        self.planned_route_stop.append("OBS")
        self.planned_route_stop.append("LDR")
        self.planned_route_stop.append("PP")
        self.planned_route_stop.append("EXIT")

        self.planned_los = []
        self.planned_los.append(None)
        self.planned_los.append(prng.exponential(MEAN_LOS_OBS))
        self.planned_los.append(prng.exponential(MEAN_LOS_LDR))
        self.planned_los.append(prng.exponential(MEAN_LOS_PP))

        # Since we have fixed route for now, just initialize full list to
        # hold bed requests
        self.bed_requests = [None for i in range(self.route_length+1)]
        self.request_entry_ts = [None for i in range(self.route_length+1)]
        self.entry_ts = [None for i in range(self.route_length + 1)]
        self.exit_ts = [None for i in range(self.route_length + 1)]

    def __repr__(self):
        return "patientid: {}, arrstream: {}, time: {}". \
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

    def __init__(self, env, arr_stream, arr_rate, obunits, initial_delay=0,
                 stoptime=simpy.core.Infinity, max_arrivals=simpy.core.Infinity, debug=False):

        self.id = id
        self.env = env
        self.arr_rate = arr_rate
        self.arr_stream = arr_stream
        self.initial_delay = initial_delay
        self.stoptime = stoptime
        self.max_arrivals = max_arrivals
        self.debug = debug
        self.out = None
        self.num_patients_created = 0

        self.prng = RandomState(RNG_SEED)

        self.action = env.process(self.run())  # starts the run() method as a SimPy process


    def run(self):
        """The patient generator.
        """
        # Delay for initial_delay
        yield self.env.timeout(self.initial_delay)
        # Main generator loop that terminates when stoptime reached
        while self.env.now < self.stoptime and self.num_patients_created<self.max_arrivals:
            # Delay until time for next arrival
            # Compute next interarrival time
            iat = self.prng.exponential(1.0 / self.arr_rate)
            yield self.env.timeout(iat)
            self.num_patients_created += 1
            # Create new patient
            obp = OBPatient(self.env.now, self.arr_stream, obunits,
                            patient_id=self.num_patients_created, prng=self.prng)
            if self.debug:
                print("Patient {} created at {:.2f}.".format(self.num_patients_created, env.now))

            self.out = obunits[obp.planned_route_stop[1]]
            self.env.process(self.out.put(obp))



# Initialize a simulation environment
env = simpy.Environment()

rho_obs = ARR_RATE * MEAN_LOS_OBS / CAPACITY_OBS
rho_ldr = ARR_RATE * MEAN_LOS_LDR / CAPACITY_LDR
rho_pp = ARR_RATE * MEAN_LOS_PP / CAPACITY_PP

print("rho_obs: {:6.3f}\nrho_ldr: {:6.3f}\nrho_pp: {:6.3f}".format(rho_obs, rho_ldr, rho_pp))

# Create nursing units
obs_unit = OBunit(env, 'OBS', CAPACITY_OBS, debug=False)
ldr_unit = OBunit(env, 'LDR', CAPACITY_LDR, debug=False)
pp_unit = OBunit(env, 'PP', CAPACITY_PP, debug=False)

# Define system exit
exitflow = ExitFlow(env, 'EXIT', store_obp=False)

# Create dictionary of units keyed by name. This object can be passed along
# to other objects so that the units are accessible as patients "flow".
obunits = {}
obunits['OBS'] = obs_unit
obunits['LDR'] = ldr_unit
obunits['PP'] = pp_unit
obunits['EXIT'] = exitflow

# Create a patient generator
obpat_gen = OBPatientGenerator(env, "Type1", ARR_RATE, obunits, 0, debug=False)

# Routing logic
# Currently routing logic is hacked into the OBPatientGenerator and OBPatient objects

# Run the simulation for a while
runtime = 10000
env.run(until=runtime)

# Patient generator stats
print("\nNum patients generated: {}\n".format(obpat_gen.num_patients_created))

# Unit stats
print(obs_unit.basic_stats_msg())
print(ldr_unit.basic_stats_msg())
print(pp_unit.basic_stats_msg())

# System exit stats
print("\nNum patients exit system: {}\n".format(exitflow.num_exits))
print("Last exit at: {:.2f}\n".format(exitflow.last_exit))

