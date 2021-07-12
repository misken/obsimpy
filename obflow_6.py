from sys import stdout
import logging
from enum import IntEnum
from copy import deepcopy
from pathlib import Path

import simpy
from numpy.random import default_rng
import networkx as nx
import pandas as pd
import yaml


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
    def __init__(self, env, locations, global_vars):

        self.env = env

        # Create individual patient care units
        # enter = EnterFlow(self.env, 'ENTRY')
        # exit = ExitFlow(self.env, 'EXIT')
        # self.obunits = [enter]

        self.obunits = []

        # Unit index in obunits list should correspond to Unit enum value
        for location in locations:
            self.obunits.append(OBunit(env, id=location, name=locations[location]['name'],
                                       capacity=locations[location]['capacity']))
        #self.obunits.append(exit)

        self.global_vars = global_vars

        # Create list to hold timestamps dictionaries (one per patient stop)
        self.stop_timestamps_list = []

        # Create list to hold timestamps dictionaries (one per patient)
        self.stops_timestamps_list = []

        # Create lists to hold occupancy tuples (sim time, occ)
        self.occupancy_lists = {}
        for location in locations:
            self.occupancy_lists[locations[location]['name']] = [(0.0, 0.0)]


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

    def __init__(self, env, id, name, capacity=simpy.core.Infinity):

        self.env = env
        self.id = id
        self.name = name
        self.capacity = capacity

        # Use a simpy Resource as one of the class members
        self.unit = simpy.Resource(env, capacity)

        # Statistical accumulators
        self.num_entries = 0
        self.num_exits = 0
        self.tot_occ_time = 0.0
        self.last_entry = None
        self.last_exit = None

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

        obpatient.current_stop_num += 1
        logger.debug(f"{obpatient.name} trying to get {self.name} at {self.env.now:.4f} for stop_num {obpatient.current_stop_num}")

        # Timestamp of request time
        bed_request_ts = self.env.now
        # Request a bed
        bed_request = self.unit.request()
        # Store bed request and timestamp in patient's request lists
        obpatient.bed_requests[obpatient.current_stop_num] = bed_request
        obpatient.unit_stops[obpatient.current_stop_num] = self.id
        obpatient.request_entry_ts[obpatient.current_stop_num] = self.env.now

        # Yield until we get a bed
        yield bed_request

        # Seized a bed.
        # Increments patient's attribute number of units visited (includes ENTRY and EXIT)

        obpatient.entry_ts[obpatient.current_stop_num] = self.env.now
        obpatient.current_unit_id = self.id

        self.num_entries += 1
        self.last_entry = self.env.now

        # Check if we have a bed from a previous stay and release it if we do.
        # Update stats for previous unit.

        if obpatient.bed_requests[obpatient.current_stop_num - 1] is not None:
            obpatient.previous_unit_id = obpatient.unit_stops[obpatient.current_stop_num - 1]
            previous_unit = obsystem.obunits[obpatient.previous_unit_id]
            previous_request = obpatient.bed_requests[obpatient.current_stop_num - 1]
            previous_unit.unit.release(previous_request)
            previous_unit.tot_occ_time += \
                self.env.now - obpatient.entry_ts[obpatient.current_stop_num - 1]
            previous_unit.num_exits += 1
            previous_unit.last_exit = self.env.now

            obpatient.request_exit_ts[obpatient.current_stop_num - 1] = self.env.now
            obpatient.request_entry_ts[obpatient.current_stop_num] = self.env.now
            obpatient.exit_ts[obpatient.current_stop_num - 1] = self.env.now

        logger.debug(f"{self.env.now:.4f}:{obpatient.name} entering {self.name} at {self.env.now:.4f}")
        logger.debug(f"{self.env.now:.4f}:{obpatient.name} waited {self.env.now - bed_request_ts:.4f} time units for {self.name} bed")

        # Retrieve los and then yield for the stay
        los = obpatient.route_graph.nodes(data=True)[obpatient.current_unit_id]['planned_los']
        obpatient.planned_los[obpatient.current_stop_num] = los
        yield self.env.timeout(los)

        # Go to next destination (which could be an exitflow)
        if obpatient.current_unit_id == Unit.EXIT:
            obpatient.previous_unit_id = obpatient.unit_stops[obpatient.current_stop_num]
            previous_unit = obsystem.obunits[obpatient.previous_unit_id]
            previous_request = obpatient.bed_requests[obpatient.current_stop_num]
            previous_unit.unit.release(previous_request)
            previous_unit.tot_occ_time += \
                self.env.now - obpatient.entry_ts[obpatient.current_stop_num]
            previous_unit.num_exits += 1
            previous_unit.last_exit = self.env.now

            obpatient.request_exit_ts[obpatient.current_stop_num] = self.env.now
            obpatient.exit_ts[obpatient.current_stop_num] = self.env.now
            self.exit_system(obpatient)
        else:
            obpatient.next_unit_id = obpatient.router.get_next_unit_id(obpatient)
            self.env.process(obsystem.obunits[obpatient.next_unit_id].put(obpatient, obsystem))

        # EXIT is now an OBunit so following is deprecated
        # if obpatient.next_unit_id == Unit.EXIT:
        #     # For ExitFlow object, no process needed
        #     obsystem.obunits[obpatient.next_unit_id].put(obpatient, obsystem)
        # else:
        #     # Process for putting patient into next bed
        #     self.env.process(obsystem.obunits[obpatient.next_unit_id].put(obpatient, obsystem))

    def exit_system(self, obpatient):

        logger.debug(f"{self.env.now:.4f}:Patient {obpatient.name} exited system at {self.env.now:.2f}.")

        # Create dictionaries of timestamps for patient_stop log
        for stop in range(len(obpatient.unit_stops)):
            if obpatient.unit_stops[stop] is not None:
                timestamps = {'patient_id': obpatient.patient_id,
                              'patient_type': obpatient.patient_type.value,
                              'unit': obpatient.unit_stops[stop],
                              'request_entry_ts': obpatient.request_entry_ts[stop],
                              'entry_ts': obpatient.entry_ts[stop],
                              'request_exit_ts': obpatient.request_exit_ts[stop],
                              'exit_ts': obpatient.exit_ts[stop],
                              'planned_los': obpatient.planned_los[stop]}

                obsystem.stop_timestamps_list.append(timestamps)


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

        logger.debug(f"{self.env.now:.4f}:{obpatient.name} entered system at {self.env.now:.4f}.")

        obpatient.current_unit_id = Unit.ENTRY

        # Go to first OB unit destination
        obpatient.next_unit_id = obpatient.router.get_next_unit_id(obpatient)

        self.env.process(obsystem.obunits[obpatient.next_unit_id].put(obpatient, obsystem))

    def basic_stats_msg(self):
        """ Create summary message with basic stats on exits.


        Returns
        -------
        str
            Message with basic stats
        """

        msg = "{:6}:\t Entries={}, Last Entry={:10.2f}".format(self.name,
                                                            self.num_entries,
                                                            self.last_entry)

        return msg

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

    def __init__(self, env, name, store_obp=False):
        self.store = simpy.Store(env)
        self.env = env
        self.name = name
        self.store_obp = store_obp

        self.num_entries = 0
        self.num_exits = 0
        self.last_exit = 0.0

    def put(self, obpatient, obsystem):

        # The following are immediately updateable since no resource needed
        self.num_entries += 1
        obpatient.current_stop_num += 1

        # Update previous, current, next unit ids
        obpatient.previous_unit_id = obpatient.current_unit_id
        obpatient.current_unit_id = Unit.EXIT
        obpatient.next_unit_id = None

        # Check if we have a bed from a previous stay and release it if we do.
        # Update stats for previous unit.

        previous_unit = obsystem.obunits[obpatient.previous_unit_id]
        previous_unit.num_exits += 1

        if obpatient.bed_requests[obpatient.current_stop_num - 1] is not None:
            previous_request = obpatient.bed_requests[obpatient.current_stop_num - 1]
            previous_unit.unit.release(previous_request)
            previous_unit.tot_occ_time += \
                self.env.now - obpatient.entry_ts[obpatient.current_stop_num - 1]
            obpatient.request_exit_ts[obpatient.current_stop_num - 1] = self.env.now
            obpatient.exit_ts[obpatient.current_stop_num - 1] = self.env.now

        # Update stats for this EXIT unit
        self.last_exit = self.env.now
        self.num_exits += 1

        logger.debug(f"{self.env.now:.4f}:Patient {obpatient.name} exited system at {self.env.now:.2f}.")

        # Create dictionaries of timestamps for patient_stop log
        for stop in range(len(obpatient.unit_stops)):
            if obpatient.unit_stops[stop] is not None:
                timestamps = {'patient_id': obpatient.patient_id,
                              'patient_type': obpatient.patient_type.value,
                              'unit': obpatient.unit_stops[stop],
                              'request_entry_ts': obpatient.request_entry_ts[stop],
                              'entry_ts': obpatient.entry_ts[stop],
                              'request_exit_ts': obpatient.request_exit_ts[stop],
                              'exit_ts': obpatient.exit_ts[stop],
                              'planned_los': obpatient.planned_los[stop]}

                obsystem.stop_timestamps_list.append(timestamps)

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
    """

    """

    def __init__(self, obsystem, router, arr_time, patient_id, arr_stream_rg):
        """

        Parameters
        ----------
        obsystem
        router
        arr_time
        patient_id
        arr_stream_rg
        """
        self.system_arrival_ts = arr_time
        self.patient_id = patient_id
        self.router = router

        # Determine patient type
        if arr_stream_rg.random() > obsystem.global_vars['c_sect_prob']:
            self.patient_type = PatientType.REG_DELIVERY_UNSCHED
        else:
            self.patient_type = PatientType.CSECT_DELIVERY_UNSCHED

        self.name = f'Patient_i{patient_id}_t{self.patient_type}'

        self.current_stop_num = 0
        self.previous_unit_id = None
        self.current_unit_id = None
        self.next_unit_id = None

        self.route_graph = router.create_route(self.patient_type)
        self.route_length = len(self.route_graph.nodes) # Includes ENTRY and EXIT

        # Since we have fixed route, just initialize full list to hold bed requests
        # The index numbers are stop numbers and so slot 0 is unused and set to None
        self.bed_requests = [None for _ in range(self.route_length + 1)]
        self.unit_stops = [None for _ in range(self.route_length + 1)]
        self.planned_los = [None for _ in range(self.route_length + 1)]
        self.request_entry_ts = [None for _ in range(self.route_length + 1)]
        self.entry_ts = [None for _ in range(self.route_length + 1)]
        self.request_exit_ts = [None for _ in range(self.route_length + 1)]
        self.exit_ts = [None for _ in range(self.route_length + 1)]

        self.system_exit_ts = None


    def __repr__(self):
        return "patientid: {}, patient_type: {}, time: {}". \
            format(self.patient_id, self.patient_type, self.arr_time)


class OBStaticRouter(object):
    def __init__(self, env, locations, routes, rg):
        """

        Parameters
        ----------
        env
        obsystem
        routes
        rg
        """

        self.env = env
        self.obsystem = obsystem
        self.rg = rg

        # List of networkx DiGraph objects. Padded with None at 0 index to align with patient type ints
        self.route_graphs = {}

        # Create route templates from routes list (of unit numbers)
        route_num = 0
        for route_num, route in routes.items():
            route_graph = nx.DiGraph()
            #route_num = route['id']

            # Add each unit number as a node
            for loc_num, location in locations.items():
                route_graph.add_node(location['id'], id=location['id'],
                                     planned_los=0.0, actual_los=0.0, blocked_duration=0.0,
                                     name=location['name'])

            # Add edges - simple serial route in this case
            for edge in route['edges']:
                route_graph.add_edge(edge['from'], edge['to'])

            # Each patient will eventually end up with their own copy of the route since it contains LOS values
            self.route_graphs[route_num] = route_graph.copy()
            logger.debug(f"{self.env.now:.4f}:route graph {route_num} - {route_graph.edges}")

    def create_route(self, patient_type):
        """

        Parameters
        ----------
        patient_type

        Returns
        -------

        """

        # Copy the route template to create new graph object
        route_graph = deepcopy(self.route_graphs[patient_type])

        # Pull out the LOS parameters for convenience
        k_obs = self.obsystem.global_vars['num_erlang_stages_obs']
        mean_los_obs = self.obsystem.global_vars['mean_los_obs']
        k_ldr = self.obsystem.global_vars['num_erlang_stages_ldr']
        mean_los_ldr = self.obsystem.global_vars['mean_los_ldr']
        k_pp = self.obsystem.global_vars['num_erlang_stages_pp']
        mean_los_pp_noc = self.obsystem.global_vars['mean_los_pp_noc']
        mean_los_pp_c = self.obsystem.global_vars['mean_los_pp_c']

        # Generate the random planned LOS values by patient type
        if patient_type == PatientType.REG_DELIVERY_UNSCHED:
            route_graph.nodes[Unit.OBS]['planned_los'] = self.rg.gamma(k_obs, mean_los_obs / k_obs)
            route_graph.nodes[Unit.LDR]['planned_los'] = self.rg.gamma(k_ldr, mean_los_ldr / k_ldr)
            route_graph.nodes[Unit.PP]['planned_los'] = self.rg.gamma(k_pp, mean_los_pp_noc / k_pp)

        elif patient_type == PatientType.CSECT_DELIVERY_UNSCHED:
            k_csect = self.obsystem.global_vars['num_erlang_stages_csect']
            mean_los_csect = self.obsystem.global_vars['mean_los_csect']

            route_graph.nodes[Unit.OBS]['planned_los'] = self.rg.gamma(k_obs, mean_los_obs / k_obs)
            route_graph.nodes[Unit.LDR]['planned_los'] = self.rg.gamma(k_ldr, mean_los_ldr / k_ldr)
            route_graph.nodes[Unit.CSECT]['planned_los'] = self.rg.gamma(k_csect, mean_los_csect / k_csect)
            route_graph.nodes[Unit.PP]['planned_los'] = self.rg.gamma(k_pp, mean_los_pp_c / k_pp)

        return route_graph

    def get_next_unit_id(self, obpatient):

        G = obpatient.route_graph
        successors = [G.nodes(data='id')[n] for n in G.successors(obpatient.current_unit_id)]
        next_unit_id = successors[0]

        if next_unit_id is None:
            logger.error(f"{self.env.now:.4f}:{obpatient.name} has no next unit at {obpatient.current_unit_id}.")
            exit(1)

        logger.debug(f"{self.env.now:.4f}:{obpatient.name} current_unit_id {obpatient.current_unit_id}, next_unit_id {next_unit_id}")
        return next_unit_id


class OBPatientGenerator(object):
    """ Generates patients.

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        obsystem : OBSystem
            the OB system containing the obunits list
        router : OBStaticRouter like
            used to route new arrival to first location
        arr_rate : float
            Poisson arrival rate (expected number of arrivals per unit time)
        arr_stream_rg : numpy.random.Generator
            used for interarrival time generation
        initial_delay : float
            Starts generation after an initial delay. (default 0.0)
        stoptime : float
            Stops generation at the stoptime. (default Infinity)
        max_arrivals : int
            Stops generation after max_arrivals. (default Infinity)

    """

    def __init__(self, env, obsystem, router, arr_rate, arr_stream_rg,
                 initial_delay=0, stoptime=simpy.core.Infinity, max_arrivals=simpy.core.Infinity):

        self.env = env
        self.obsystem = obsystem
        self.router = router
        self.arr_rate = arr_rate
        self.arr_stream_rg = arr_stream_rg
        self.initial_delay = initial_delay
        self.stoptime = stoptime
        self.max_arrivals = max_arrivals

        self.out = None
        self.num_patients_created = 0

        # Register the run() method as a SimPy process
        env.process(self.run())

    def run(self):
        """The patient generator.
        """

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

            logger.debug(f"{self.env.now:.4f}:Patient {obpatient.name} created at {self.env.now:.4f}.")

            # Initiate process of patient entering system
            self.env.process(self.obsystem.obunits[Unit.ENTRY].put(obpatient, self.obsystem))


def simulate(arg_dict, rep_num):
    """

    Parameters
    ----------
    arg_dict : dict whose keys are the input args
    rep_num : int, simulation replication number

    Returns
    -------
    Nothing returned but numerous output files written to ``args_dict[output_path]``

    """

    # Create a random number generator for this replication
    seed = arg_dict['seed'] + rep_num - 1
    rg = default_rng(seed=seed)

    # Resource capacity levels
    num_greeters = arg_dict['num_greeters']
    num_reg_staff = arg_dict['num_reg_staff']
    num_vaccinators = arg_dict['num_vaccinators']
    num_schedulers = arg_dict['num_schedulers']

    # Initialize the patient flow related attributes
    patient_arrival_rate = arg_dict['patient_arrival_rate']
    mean_interarrival_time = 1.0 / (patient_arrival_rate / 60.0)

    pct_need_second_dose = arg_dict['pct_need_second_dose']
    temp_check_time_mean = arg_dict['temp_check_time_mean']
    temp_check_time_sd = arg_dict['temp_check_time_sd']
    reg_time_mean = arg_dict['reg_time_mean']
    vaccinate_time_mean = arg_dict['vaccinate_time_mean']
    vaccinate_time_sd = arg_dict['vaccinate_time_sd']
    sched_time_mean = arg_dict['sched_time_mean']
    sched_time_sd = arg_dict['sched_time_sd']
    obs_time = arg_dict['obs_time']
    post_obs_time_mean = arg_dict['post_obs_time_mean']

    # Other parameters
    stoptime = arg_dict['stoptime']  # No more arrivals after this time
    quiet = arg_dict['quiet']
    scenario = arg_dict['scenario']

    # Run the simulation
    env = simpy.Environment()

    # Create a clinic to simulate
    clinic = VaccineClinic(env, num_greeters, num_reg_staff, num_vaccinators, num_schedulers,
                           pct_need_second_dose,
                           temp_check_time_mean, temp_check_time_sd,
                           reg_time_mean,
                           vaccinate_time_mean, vaccinate_time_sd,
                           sched_time_mean, sched_time_sd,
                           obs_time, post_obs_time_mean, rg
                           )

    # Initialize and register (happens in __init__) the patient arrival generators
    walkin_gen = WalkinPatientGenerator(env, clinic, mean_interarrival_time, stoptime, rg, quiet=quiet)
    scheduled_gen = ScheduledPatientGenerator(env, clinic, 10.0, 5, stoptime, rg, quiet=quiet)

    # Launch the simulation
    env.run()

    # Create output files and basic summary stats
    if len(arg_dict['output_path']) > 0:
        output_dir = Path.cwd() / arg_dict['output_path']
    else:
        output_dir = Path.cwd()

    # Create paths for the output logs
    clinic_patient_log_path = output_dir / f'clinic_patient_log_{scenario}_{rep_num}.csv'
    vac_occupancy_df_path = output_dir / f'vac_occupancy_{scenario}_{rep_num}.csv'
    postvac_occupancy_df_path = output_dir / f'postvac_occupancy_{scenario}_{rep_num}.csv'

    # Create patient log dataframe and add scenario and rep number cols
    clinic_patient_log_df = pd.DataFrame(clinic.timestamps_list)
    clinic_patient_log_df['scenario'] = scenario
    clinic_patient_log_df['rep_num'] = rep_num

    # Reorder cols to get scenario and rep_num first
    num_cols = len(clinic_patient_log_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols - 2)])
    clinic_patient_log_df = clinic_patient_log_df.iloc[:, new_col_order]

    # Compute durations of interest for patient log
    clinic_patient_log_df = compute_durations(clinic_patient_log_df)

    # Create occupancy log dataframes and add scenario and rep number cols
    vac_occupancy_df = pd.DataFrame(clinic.vac_occupancy_list, columns=['ts', 'occ'])
    vac_occupancy_df['scenario'] = scenario
    vac_occupancy_df['rep_num'] = scenario
    num_cols = len(vac_occupancy_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols - 2)])
    vac_occupancy_df = vac_occupancy_df.iloc[:, new_col_order]

    postvac_occupancy_df = pd.DataFrame(clinic.postvac_occupancy_list, columns=['ts', 'occ'])
    postvac_occupancy_df['scenario'] = scenario
    postvac_occupancy_df['rep_num'] = scenario
    num_cols = len(postvac_occupancy_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols - 2)])
    postvac_occupancy_df = postvac_occupancy_df.iloc[:, new_col_order]

    # Export logs to csv
    clinic_patient_log_df.to_csv(clinic_patient_log_path, index=False)
    # vac_occupancy_df.to_csv(vac_occupancy_df_path, index=False)
    # postvac_occupancy_df.to_csv(postvac_occupancy_df_path, index=False)

    # Note simulation end time
    end_time = env.now
    print(f"Simulation replication {rep_num} ended at time {end_time}")

def process_command_line():
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='vaccine_clinic_model4',
                                     description='Run vaccine clinic simulation')

    # Add arguments
    parser.add_argument(
        "--config", type=str, default=None,
        help="Configuration file containing input parameter arguments and values"
    )

    parser.add_argument("--patient_arrival_rate", default=150, help="patients per hour",
                        type=float)

    parser.add_argument("--num_greeters", default=2, help="number of greeters",
                        type=int)

    parser.add_argument("--num_reg_staff", default=2, help="number of registration staff",
                        type=int)

    parser.add_argument("--num_vaccinators", default=15, help="number of vaccinators",
                        type=int)

    parser.add_argument("--num_schedulers", default=2, help="number of schedulers",
                        type=int)

    parser.add_argument("--pct_need_second_dose", default=0.5,
                        help="percent of patients needing 2nd dose (default = 0.5)",
                        type=float)

    parser.add_argument("--temp_check_time_mean", default=0.25,
                        help="Mean time (mins) for temperature check (default = 0.25)",
                        type=float)

    parser.add_argument("--temp_check_time_sd", default=0.05,
                        help="Standard deviation time (mins) for temperature check (default = 0.05)",
                        type=float)

    parser.add_argument("--reg_time_mean", default=1.0,
                        help="Mean time (mins) for registration (default = 1.0)",
                        type=float)

    parser.add_argument("--vaccinate_time_mean", default=4.0,
                        help="Mean time (mins) for vaccination (default = 4.0)",
                        type=float)

    parser.add_argument("--vaccinate_time_sd", default=0.5,
                        help="Standard deviation time (mins) for vaccination (default = 0.5)",
                        type=float)

    parser.add_argument("--sched_time_mean", default=1.0,
                        help="Mean time (mins) for scheduling 2nd dose (default = 1.0)",
                        type=float)

    parser.add_argument("--sched_time_sd", default=1.0,
                        help="Standard deviation time (mins) for scheduling 2nd dose (default = 0.1)",
                        type=float)

    parser.add_argument("--obs_time", default=15.0,
                        help="Time (minutes) patient waits post-vaccination in observation area (default = 15)",
                        type=float)

    parser.add_argument("--post_obs_time_mean", default=1.0,
                        help="Time (minutes) patient waits post OBS_TIME in observation area (default = 1.0)",
                        type=float)

    parser.add_argument(
        "--scenario", type=str, default=datetime.now().strftime("%Y.%m.%d.%H.%M."),
        help="Appended to output filenames."
    )

    parser.add_argument("--stoptime", default=600, help="time that simulation stops (default = 600)",
                        type=float)

    parser.add_argument("--num_reps", default=1, help="number of simulation replications (default = 1)",
                        type=int)

    parser.add_argument("--seed", default=3, help="random number generator seed (default = 3)",
                        type=int)

    parser.add_argument(
        "--output_path", type=str, default="", help="location for output file writing")

    parser.add_argument("--quiet", action='store_true',
                        help="If True, suppresses output messages (default=False")

    # do the parsing
    args = parser.parse_args()

    if args.config is not None:
        # Read inputs from config file
        with open(args.config, "r") as fin:
            args = parser.parse_args(fin.read().split())

    return args


def main():
    args = process_command_line()
    print(args)

    num_reps = args.num_reps
    scenario = args.scenario

    if len(args.output_path) > 0:
        output_dir = Path.cwd() / args.output_path
    else:
        output_dir = Path.cwd()

    # Main simulation replication loop
    for i in range(1, num_reps + 1):
        simulate(vars(args), i)

    # Consolidate the patient logs and compute summary stats
    patient_log_stats = process_sim_output(output_dir, scenario)
    print(f"\nScenario: {scenario}")
    pd.set_option("display.precision", 3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(patient_log_stats['patient_log_rep_stats'])
    print(patient_log_stats['patient_log_ci'])


if __name__ == '__main__':

    # Main program

    loglevel = 'DEBUG' # TODO - from file or input arg

    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=stdout,
    )

    logger = logging.getLogger(__name__)

    with open('test_config.yaml', 'rt') as yamlfile:
        inputs = yaml.safe_load(yamlfile)

    global_vars = inputs['global_vars']
    random_number_streams = inputs['random_number_streams']
    locations = inputs['locations']
    routes = inputs['routes']


    # Initialize a simulation environment
    env = simpy.Environment()


    # global_vars = {
    #     'arrival_rate': 0.4,
    #     'mean_los_obs': 3.0,
    #     'num_erlang_stages_obs': 4,
    #     'mean_los_ldr': 12.0,
    #     'num_erlang_stages_ldr': 4,
    #     'mean_los_pp_c': 72.0,
    #     'mean_los_pp_noc': 48.0,
    #     'num_erlang_stages_pp': 4,
    #     'mean_los_csect': 1,
    #     'num_erlang_stages_csect': 4,
    #     'c_sect_prob': 0.00
    # }

    # random_number_streams = {
    #     'arrivals': 27,
    #     'los': 19
    # }

    # Units - this spec should be read from a YAML or JSON input file

    # obunits_list = [{'name': 'OBS', 'capacity': 100},
    #                 {'name': 'LDR', 'capacity': 100},
    #                 {'name': 'CSECT', 'capacity': 100},
    #                 {'name': 'PP', 'capacity': 100}]


    # Create an OB System
    obsystem = OBsystem(env, locations, global_vars)

    # Compute and display traffic intensities
    rho_obs = global_vars['arrival_rate'] * global_vars['mean_los_obs'] / locations[Unit.OBS]['capacity']
    rho_ldr = global_vars['arrival_rate'] * global_vars['mean_los_ldr'] / locations[Unit.LDR]['capacity']
    mean_los_pp = global_vars['mean_los_pp_c'] * global_vars['c_sect_prob'] + \
        global_vars['mean_los_pp_noc'] * (1 - global_vars['c_sect_prob'])
    rho_pp = global_vars['arrival_rate'] * mean_los_pp / locations[Unit.PP]['capacity']

    print(f"rho_obs: {rho_obs:6.3f}\nrho_ldr: {rho_ldr:6.3f}\nrho_pp: {rho_pp:6.3f}")

    # Create random number generators
    rg = {}
    for stream in random_number_streams:
        rg[stream] = default_rng(random_number_streams[stream])

    # Create a router - hard coded lists for now (TODO - read in from file or input arg)
    # route_1_units = [Unit.ENTRY, Unit.OBS, Unit.LDR, Unit.PP, Unit.EXIT]
    # route_2_units = [Unit.ENTRY, Unit.OBS, Unit.LDR, Unit.CSECT, Unit.PP, Unit.EXIT]
    # routes = [route_1_units, route_2_units]
    router = OBStaticRouter(env, locations, routes, rg['los'])

    # Create a patient generator
    obpat_gen = OBPatientGenerator(env, obsystem, router, global_vars['arrival_rate'],
                                   rg['arrivals'], max_arrivals=10000)

    # Run the simulation for a while (TODO - read in from file or input arg)
    runtime = 10000
    env.run(until=runtime)

    # for ts_list in obsystem.stop_timestamps_list:
    #     print(ts_list)

    # Patient generator stats
    print("\nNum patients generated: {}\n".format(obpat_gen.num_patients_created))

    # Unit stats
    for unit in obsystem.obunits[1:-1]:
        print(unit.basic_stats_msg())

    # System exit stats

    print("\nNum patients exiting system: {}\n".format(obsystem.obunits[Unit.EXIT].num_exits))
    print("Last exit at: {:.2f}\n".format(obsystem.obunits[Unit.EXIT].last_exit))

