from sys import stdout
import logging
from enum import IntEnum

import simpy
from numpy.random import default_rng
import networkx as nx


"""
Simple OB patient flow model 6 - Very simple OO

Details:

- Generate arrivals via Poisson process
- Define an OBUnit class that contains a simpy.Resource object as a member.
  Not subclassing Resource, just trying to use it as a member.
- Routing is done via setting ``out`` member of an OBUnit instance to
 another OBUnit instance to which the OB patient flow instance should be
 routed. The routing logic, for now, is in OBUnit object. Later,
 we need some sort of router object and data driven routing.
- Trying to get patient flow working without a process function that
explicitly articulates the sequence of units and stays.

Planned enhancements from obflow_5:

Goal is to be able to run new scenarios for the obsim experiments. Model
needs to match Simio model functionality.

- LOS distributions that match the obsim experiments
- LOS adjustment in LDR based on wait time in OBS
- logging
- read key scenario inputs from a file

Key Lessons Learned:

- Any function that is a generator and might potentially yield for an event
  must get registered as a process.

"""



class OBsystem(object):
    def __init__(self, env, obunits, global_vars):

        self.env = env

        # Create individual patient care units
        enter = EnterFlow(self.env, 'Entry')
        exit = ExitFlow(self.env, 'Exit')
        self.obunits = [enter]
        # Unit index in obunits list should correspond to Unit enum value
        for unit in obunits:
            self.obunits.append(OBunit(env, name=unit['name'], capacity=unit['capacity']))
        self.obunits.append(exit)

        self.global_vars = global_vars

        # Create list to hold timestamps dictionaries (one per patient)
        self.patient_timestamps_list = []

        # Create lists to hold occupancy tuples (time, occ)
        self.occupancy_lists = {}
        for unit in obunits:
            self.occupancy_lists[unit['name']] = [(0.0, 0.0)]


class PatientType(IntEnum):
    REG_DELIVERY_UNSCHED = 1
    CSECT_DELIVERY_UNSCHED = 2

class Unit(IntEnum):
    ENTRY = 0
    OBS = 1
    LDR = 2
    CSECT = 3
    PP = 4
    EXIT = 5


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

    def __init__(self, env, name, capacity=simpy.core.Infinity):

        self.env = env
        self.name = name
        self.capacity = capacity

        # Use a simpy Resource as one of the class members
        self.unit = simpy.Resource(env, capacity)

        # Statistical accumulators
        self.num_entries = 0
        self.num_exits = 0
        self.tot_occ_time = 0.0

    def put(self, obpatient, obsystem):
        """ A process method called when a bed is requested in the unit.

            The logic of this method is reminiscent of the routing logic
            in the process oriented obflow models 1-3. However, this method
            is used for all of the units - no repeated logic.

            Parameters
            ----------
            env : simpy.Environment
                the simulation environment

            obpatient : OBPatient object
                the patient requesting the bed

        """

        logger.debug(f"{obpatient.name} trying to get {self.name} at {self.env.now:.4f}")

        # Increments patient's attribute number of units visited
        obpatient.current_stay_num += 1
        # Timestamp of request time
        bed_request_ts = self.env.now
        # Request a bed
        bed_request = self.unit.request()
        # Store bed request and timestamp in patient's request lists
        obpatient.bed_requests[obpatient.current_stay_num] = bed_request
        obpatient.request_entry_ts[obpatient.current_stay_num] = self.env.now

        # Yield until we get a bed
        yield bed_request

        # Seized a bed.
        obpatient.entry_ts[obpatient.current_stay_num] = self.env.now
        obpatient.previous_unit_id = obpatient.current_unit_id
        obpatient.current_unit_id = obpatient.next_unit_id
        obpatient.next_unit_id = None

        # Check if we have a bed from a previous stay and release it.
        # Update stats for previous unit.

        if obpatient.bed_requests[obpatient.current_stay_num - 1] is not None:
            previous_request = obpatient.bed_requests[obpatient.current_stay_num - 1]

            previous_unit = obsystem.obunits[obpatient.previous_unit_id]
            previous_unit.unit.release(previous_request)
            previous_unit.num_exits += 1
            previous_unit.tot_occ_time += \
                self.env.now - obpatient.entry_ts[obpatient.current_stay_num - 1]
            obpatient.exit_ts[obpatient.current_stay_num - 1] = self.env.now

        logger.debug(f"{obpatient.name} entering {self.name} at {self.env.now:.4f}")

        self.num_entries += 1
        logger.debug(f"{obpatient.name} waited {self.env.now - bed_request_ts:.4f} time units for {self.name} bed")

        # Determine los and then yield for the stay
        los = obpatient.route_graph.nodes(data=True)[obpatient.current_unit_id]['planned_los']
        yield self.env.timeout(los)

        # Go to next destination (which could be an exitflow)
        next_unit_id = obpatient.router.get_next_unit_id(obpatient)
        obpatient.next_unit_id = next_unit_id

        if obpatient.next_unit_id == Unit.EXIT:
            # For ExitFlow object, no process needed
            obsystem.obunits[obpatient.next_unit_id].put(obpatient, obsystem)
        else:
            # Process for putting patient into next bed
            self.env.process(obsystem.obunits[obpatient.next_unit_id].put(obpatient, obsystem))


    def basic_stats_msg(self):
        """ Compute entries, exits, avg los and create summary message.


        Returns
        -------
        str
            Message with basic stats
        """

        if self.num_exits > 0:
            alos = self.tot_occ_time / self.num_exits
        else:
            alos = 0

        msg = "{:6}:\t Entries={}, Exits={}, Occ={}, ALOS={:4.2f}".\
            format(self.name, self.num_entries, self.num_exits,
                   self.unit.count, alos)
        return msg


class EnterFlow(object):
    """Patients routed here upon creation"""

    def __init__(self, env, name):
        self.env = env
        self.name = name

        self.num_entries = 0
        self.num_exits = 0
        self.last_entry = None

    def put(self, obpatient, obsystem):

        self.last_entry = self.env.now
        self.num_entries += 1

        logger.debug(f"{obpatient.name} entered system at {self.env.now:.4f}.")

        obpatient.current_unit_id = Unit.ENTRY

        # Go to first OB unit destination
        next_unit_id = obpatient.router.get_next_unit_id(obpatient)
        obpatient.next_unit_id = next_unit_id

        self.num_exits += 1
        self.env.process(obsystem.obunits[next_unit_id].put(obpatient, obsystem))



class ExitFlow(object):
    """ Patients routed here when ready to exit.

        Patient objects put into a Store. Can be accessed later for stats
        and logs. A little worried about how big the Store will get.

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
        self.num_entries = 0
        self.num_exits = 0
        self.last_exit = 0.0

    def put(self, obpatient, obsystem):

        self.num_entries += 1
        obpatient.previous_unit_id = obpatient.current_unit_id
        obpatient.current_unit_id = Unit.EXIT
        obpatient.next_unit_id = None

        if obpatient.bed_requests[obpatient.current_stay_num] is not None:
            previous_request = obpatient.bed_requests[obpatient.current_stay_num]

            previous_unit = obsystem.obunits[obpatient.previous_unit_id]
            previous_unit.unit.release(previous_request)
            previous_unit.num_exits += 1
            previous_unit.tot_occ_time += \
                self.env.now - obpatient.entry_ts[obpatient.current_stay_num]
            obpatient.exit_ts[obpatient.current_stay_num] = self.env.now

        self.last_exit = self.env.now
        self.num_exits += 1

        logger.debug(f"Patient {obpatient.name} exited system at {self.env.now:.2f}.")

        # Store patient
        if self.store_obp:
            self.store.put(obpatient)

    def basic_stats_msg(self):
        """ Create summary message with basic stats on exits.


        Returns
        -------
        str
            Message with basic stats
        """

        msg = "{:6}:\t Exits={}, Last Exit={:10.2f}".format(self.name,
                                                            self.num_exits,
                                                            self.last_exit)

        return msg


class OBPatient(object):
    """ Models an OB patient

        Parameters
        ----------
        arr_time : float
            Patient arrival time
        patient_id : int
            Unique patient id
        arr_stream : int
            Arrival stream id (default 1). Currently there is just one arrival
            stream corresponding to the one patient generator class. In future,
            likely to be be multiple generators for generating random and
            scheduled arrivals.

    """

    def __init__(self, obsystem, router, arr_time, patient_id, arr_stream_rg):
        self.arr_time = arr_time
        self.patient_id = patient_id
        self.router = router
        self.current_stay_num = 0

        # Determine patient type
        if arr_stream_rg.random() > obsystem.global_vars['c_sect_prob']:
            self.patient_type = PatientType.REG_DELIVERY_UNSCHED
        else:
            self.patient_type = PatientType.CSECT_DELIVERY_UNSCHED

        self.name = f'Patient_{patient_id}_{self.patient_type}'

        self.previous_unit_id = None
        self.current_unit_id = None
        self.next_unit_id = None

        router.create_route(self)




    def __repr__(self):
        return "patientid: {}, arr_stream: {}, time: {}". \
            format(self.patient_id, self.arr_stream, self.arr_time)


class OBStaticRouter(object):
    def __init__(self, env, obsystem, routes, rg):
        self.env = env
        self.obsystem = obsystem
        self.rg = rg

        self.route_graphs = [None]

        # Create route templates from routes list (of unit numbers)
        for route in routes:
            route_graph = nx.DiGraph()

            for unit in route:
                route_graph.add_node(unit, id=unit, planned_los=0.0, actual_los=0.0, blocked_duration=0.0,
                             name=obsystem.obunits[unit].name)

            # Add edges
            for stopnum in range(0, len(route) - 1):
                route_graph.add_edge(route[stopnum], route[stopnum + 1])

            print(route_graph.edges)
            self.route_graphs.append(route_graph.copy())


    def create_route(self, obpatient):
        # Hard coding route, los and bed requests for now
        # Not sure how best to do routing related data structures.
        # Hack for now using combination of lists here, the out member
        # and the obunits dictionary.

        # Copy the route
        obpatient.route_graph = self.route_graphs[obpatient.patient_type]
        obpatient.route_length = len(obpatient.route_graph.nodes)

        k_obs = self.obsystem.global_vars['num_erlang_stages_obs']
        mean_los_obs = self.obsystem.global_vars['mean_los_obs']
        k_ldr = self.obsystem.global_vars['num_erlang_stages_ldr']
        mean_los_ldr = self.obsystem.global_vars['mean_los_ldr']
        k_pp = self.obsystem.global_vars['num_erlang_stages_pp']
        mean_los_pp_noc = self.obsystem.global_vars['mean_los_pp_noc']
        mean_los_pp_c = self.obsystem.global_vars['mean_los_pp_c']

        if obpatient.patient_type == PatientType.REG_DELIVERY_UNSCHED:
            obpatient.route_graph.nodes[Unit.OBS]['planned_los'] = self.rg.gamma(k_obs, mean_los_obs / k_obs)
            obpatient.route_graph.nodes[Unit.LDR]['planned_los'] = self.rg.gamma(k_ldr, mean_los_ldr / k_ldr)
            obpatient.route_graph.nodes[Unit.PP]['planned_los'] = self.rg.gamma(k_pp, mean_los_pp_noc / k_pp)

        elif obpatient.patient_type == PatientType.CSECT_DELIVERY_UNSCHED:
            k_csect = self.obsystem.global_vars['num_erlang_stages_csect']
            mean_los_csect = self.obsystem.global_vars['mean_los_csect']

            obpatient.route_graph.nodes[Unit.OBS]['planned_los'] = self.rg.exponential(mean_los_obs)
            obpatient.route_graph.nodes[Unit.LDR]['planned_los'] = self.rg.gamma(k_ldr, mean_los_ldr)
            obpatient.route_graph.nodes[Unit.CSECT]['planned_los'] = self.rg.gamma(k_csect, mean_los_csect)
            obpatient.route_graph.nodes[Unit.PP]['planned_los'] = self.rg.gamma(k_pp, mean_los_pp_c)

        # Since we have fixed route, just initialize full list to hold bed requests
        obpatient.bed_requests = [None for _ in range(obpatient.route_length)]
        obpatient.request_entry_ts = [None for _ in range(obpatient.route_length)]
        obpatient.entry_ts = [None for _ in range(obpatient.route_length)]
        obpatient.exit_ts = [None for _ in range(obpatient.route_length)]

    def get_next_unit_id(self, obpatient):

        G = obpatient.route_graph
        next_unit_id = \
            [G.nodes(data='id')[n] for n in G.successors(obpatient.current_unit_id)][0]

        if next_unit_id is None:
            logger.debug(f"{obpatient.name} has no next unit at {obpatient.current_unit_id}.")
            exit(1)

        print(f"{obpatient.name} current_unit_id {obpatient.current_unit_id}, next_unit_id {next_unit_id}")
        return next_unit_id


class OBPatientGenerator(object):
    """ Generates patients.

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        arr_rate : float
            Poisson arrival rate (expected number of arrivals per unit time)
        arr_stream : int
            Arrival stream id (default 0). Currently there is just one arrival
            stream corresponding to the one patient generator class. In future,
            likely to be be multiple generators for generating random and
            scheduled arrivals
        initial_delay : float
            Starts generation after an initial delay. (default 0.0)
        stoptime : float
            Stops generation at the stoptime. (default Infinity)
        max_arrivals : int
            Stops generation after max_arrivals. (default Infinity)
        rg : Generator (numpy.random), default=None
            If None, a new default_rng is created using seed
        seed : int, default=None
            Random number seed

    """

    def __init__(self, env, obsystem, router, arr_rate, arr_stream_rg,
                 initial_delay=0,
                 stoptime=simpy.core.Infinity,
                 max_arrivals=simpy.core.Infinity):

        self.obsystem = obsystem
        self.router = router
        self.env = env
        self.arr_rate = arr_rate
        self.arr_stream_rg = arr_stream_rg
        self.initial_delay = initial_delay
        self.stoptime = stoptime
        self.max_arrivals = max_arrivals

        # Register the run() method as a SimPy process
        env.process(self.run())

    def run(self):
        """The patient generator.
        """
        self.out = None
        self.num_patients_created = 0

        # Delay for initial_delay
        yield self.env.timeout(self.initial_delay)
        # Main generator loop that terminates when stoptime reached
        while self.env.now < self.stoptime and \
                        self.num_patients_created < self.max_arrivals:
            # Compute next interarrival time
            iat = self.arr_stream_rg.exponential(1.0 / self.arr_rate)
            # Delay until time for next arrival
            yield self.env.timeout(iat)
            self.num_patients_created += 1
            # Create new patient
            obpatient = OBPatient(self.obsystem, self.router, self.env.now,
                                  self.num_patients_created, self.arr_stream_rg)

            logger.debug(f"Patient {obpatient.name} created at {self.env.now:.4f}.")

            # Initiate process of patient entering system - NOT a Simpy process
            # Just using the put() pattern to be consistent using true Simpy resources
            self.obsystem.obunits[0].put(obpatient, self.obsystem)





# Logging
# Logger - TODO

loglevel = 'DEBUG' # We would get this from command line

numeric_level = getattr(logging, loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)

logging.basicConfig(
    level=numeric_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=stdout,
)

logger = logging.getLogger(__name__)


# Initialize a simulation environment
env = simpy.Environment()


global_vars = {
    'arrival_rate': 0.4,
    'mean_los_obs': 3.0,
    'num_erlang_stages_obs': 4,
    'mean_los_ldr': 12.0,
    'num_erlang_stages_ldr': 4,
    'mean_los_pp_c': 72.0,
    'mean_los_pp_noc': 48.0,
    'num_erlang_stages_pp': 4,
    'mean_los_csect': 1,
    'num_erlang_stages_csect': 4,
    'c_sect_prob': 0.00
}

random_number_streams = {
    'arrivals': 27,
    'los': 19
}

# Units - this spec should be read from a YAML or JSON input file

obunits_dict = {'OBS': {'capacity': 10},
                'LDR': {'capacity': 6},
                'CSECT': {'capacity': 6},
                'PP': {'capacity': 24}}



obunits_list = [{'name': 'OBS', 'capacity': 100},
                {'name': 'LDR', 'capacity': 100},
                {'name': 'CSECT', 'capacity': 100},
                {'name': 'PP', 'capacity': 100}]


# Create an OB System
obsystem = OBsystem(env, obunits_list, global_vars)

# Compute and display traffic intensities
rho_obs = global_vars['arrival_rate'] * global_vars['mean_los_obs'] / obunits_list[0]['capacity']
rho_ldr = global_vars['arrival_rate'] * global_vars['mean_los_ldr'] / obunits_list[1]['capacity']
mean_los_pp = global_vars['mean_los_pp_c'] * global_vars['c_sect_prob'] + \
    global_vars['mean_los_pp_noc'] * (1 - global_vars['c_sect_prob'])
rho_pp = global_vars['arrival_rate'] * mean_los_pp / obunits_list[3]['capacity']

print(f"rho_obs: {rho_obs:6.3f}\nrho_ldr: {rho_ldr:6.3f}\nrho_pp: {rho_pp:6.3f}")

# Create random number generators
rg = {}
for stream in random_number_streams:
    rg[stream] = default_rng(random_number_streams[stream])

# Create a router
route_1_units = [Unit.ENTRY, Unit.OBS, Unit.LDR, Unit.PP, Unit.EXIT]
route_2_units = [Unit.ENTRY, Unit.OBS, Unit.LDR, Unit.CSECT, Unit.PP, Unit.EXIT]
routes = [route_1_units, route_2_units]
router = OBStaticRouter(env, obsystem, routes, rg['los'])

# Create a patient generator
obpat_gen = OBPatientGenerator(env, obsystem, router, global_vars['arrival_rate'], rg['arrivals'])

# Run the simulation for a while
runtime = 10000
env.run(until=runtime)

# Patient generator stats
print("\nNum patients generated: {}\n".format(obpat_gen.num_patients_created))

# Unit stats
for unit in obsystem.obunits[1:-1]:
    print(unit.basic_stats_msg())


# System exit stats
print("\nNum patients exiting system: {}\n".format(obsystem.obunits[-1].num_exits))
print("Last exit at: {:.2f}\n".format(obsystem.obunits[-1].last_exit))
