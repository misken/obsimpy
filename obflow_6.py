from sys import stdout
import logging
from enum import IntEnum
from copy import deepcopy
from pathlib import Path
import argparse

import simpy
from numpy.random import default_rng
import networkx as nx
import pandas as pd
import yaml

from obflow_6_output import compute_occ_stats

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
            self.obunits.append(OBunit(env, unit_id=location, name=locations[location]['name'],
                                       capacity=locations[location]['capacity']))
        # self.obunits.append(exit)

        self.global_vars = global_vars

        # Create list to hold timestamps dictionaries (one per patient stop)
        self.patient_timestamps_list = []

        # Create list to hold timestamps dictionaries (one per patient)
        self.stops_timestamps_list = []


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

    def __init__(self, env, unit_id, name, capacity=simpy.core.Infinity):

        self.env = env
        self.id = unit_id
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

        # Create list to hold occupancy tuples (time, occ)
        self.occupancy_list = [(0.0, 0.0)]

    def put(self, obpatient, obsystem):
        """ A process method called when a bed is requested in the unit.

            The logic of this method is reminiscent of the routing logic
            in the process oriented obflow models 1-3. However, this method
            is used for all of the units - no repeated logic.

            Parameters
            ----------
            obpatient : OBPatient object
                the patient requesting the bed
            obsystem : OBSystem object

        """

        obpatient.current_stop_num += 1
        logger.debug(
            f"{obpatient.name} trying to get {self.name} at {self.env.now:.4f} for stop_num {obpatient.current_stop_num}")

        # Timestamp of request time
        bed_request_ts = self.env.now
        # Request a bed
        bed_request = self.unit.request()
        # Store bed request and timestamp in patient's request lists
        obpatient.bed_requests[obpatient.current_stop_num] = bed_request
        obpatient.unit_stops[obpatient.current_stop_num] = self.id
        obpatient.request_entry_ts[obpatient.current_stop_num] = self.env.now

        # If we are coming from upstream unit, we are trying to exit that unit now
        if obpatient.bed_requests[obpatient.current_stop_num - 1] is not None:
            obpatient.request_exit_ts[obpatient.current_stop_num - 1] = self.env.now

        # Yield until we get a bed
        yield bed_request

        # Seized a bed.
        # Increments patient's attribute number of units visited (includes ENTRY and EXIT)

        obpatient.entry_ts[obpatient.current_stop_num] = self.env.now
        obpatient.wait_to_enter[obpatient.current_stop_num] = self.env.now - obpatient.request_entry_ts[
            obpatient.current_stop_num]
        obpatient.current_unit_id = self.id

        self.num_entries += 1
        self.last_entry = self.env.now

        # Increment occupancy
        self.inc_occ()

        # Check if we have a bed from a previous stay and release it if we do.
        # Update stats for previous unit.

        if obpatient.bed_requests[obpatient.current_stop_num - 1] is not None:
            obpatient.exit_ts[obpatient.current_stop_num - 1] = self.env.now
            obpatient.wait_to_exit[obpatient.current_stop_num - 1] = \
                self.env.now - obpatient.request_exit_ts[obpatient.current_stop_num - 1]
            obpatient.previous_unit_id = obpatient.unit_stops[obpatient.current_stop_num - 1]
            previous_unit = obsystem.obunits[obpatient.previous_unit_id]
            previous_request = obpatient.bed_requests[obpatient.current_stop_num - 1]
            previous_unit.unit.release(previous_request)
            previous_unit.tot_occ_time += \
                self.env.now - obpatient.entry_ts[obpatient.current_stop_num - 1]
            previous_unit.num_exits += 1
            previous_unit.last_exit = self.env.now
            # Decrement occupancy
            previous_unit.dec_occ()

        logger.debug(f"{self.env.now:.4f}:{obpatient.name} entering {self.name} at {self.env.now:.4f}")
        logger.debug(
            f"{self.env.now:.4f}:{obpatient.name} waited {self.env.now - bed_request_ts:.4f} time units for {self.name} bed")

        # Retrieve los and then yield for the stay
        los = obpatient.route_graph.nodes(data=True)[obpatient.current_unit_id]['planned_los']
        obpatient.planned_los[obpatient.current_stop_num] = los

        # Do any blocking related los adjustments
        if self.name == 'LDR':
            adj_los = max(0, los - obpatient.wait_to_exit[obpatient.current_stop_num - 1])
        else:
            adj_los = los

        obpatient.adjusted_los[obpatient.current_stop_num] = adj_los

        # Wait for LOS to elapse
        yield self.env.timeout(adj_los)

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
            # Decrement occupancy
            previous_unit.dec_occ()

            obpatient.request_exit_ts[obpatient.current_stop_num] = self.env.now
            obpatient.exit_ts[obpatient.current_stop_num] = self.env.now
            self.exit_system(obpatient, obsystem)
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

    def inc_occ(self, increment=1):

        # Update occupancy - increment by 1
        prev_occ = self.occupancy_list[-1][1]
        new_occ = (self.env.now, prev_occ + increment)
        self.occupancy_list.append(new_occ)

    def dec_occ(self, decrement=1):

        # Update occupancy - increment by 1
        prev_occ = self.occupancy_list[-1][1]
        new_occ = (self.env.now, prev_occ - decrement)
        self.occupancy_list.append(new_occ)

    def exit_system(self, obpatient, obsystem):

        logger.debug(f"{self.env.now:.4f}:Patient {obpatient.name} exited system at {self.env.now:.2f}.")

        # Create dictionaries of timestamps for patient_stop log
        for stop in range(len(obpatient.unit_stops)):
            if obpatient.unit_stops[stop] is not None:
                timestamps = {'patient_id': obpatient.patient_id,
                              'patient_type': obpatient.patient_type.value,
                              'unit': Unit(obpatient.unit_stops[stop]).name,
                              'request_entry_ts': obpatient.request_entry_ts[stop],
                              'entry_ts': obpatient.entry_ts[stop],
                              'request_exit_ts': obpatient.request_exit_ts[stop],
                              'exit_ts': obpatient.exit_ts[stop],
                              'planned_los': obpatient.planned_los[stop],
                              'adjusted_los': obpatient.adjusted_los[stop],
                              'entry_tryentry': obpatient.entry_ts[stop] - obpatient.request_entry_ts[stop],
                              'tryexit_entry': obpatient.request_exit_ts[stop] - obpatient.entry_ts[stop],
                              'exit_tryexit': obpatient.exit_ts[stop] - obpatient.request_exit_ts[stop],
                              'exit_enter': obpatient.exit_ts[stop] - obpatient.entry_ts[stop],
                              'exit_tryenter': obpatient.exit_ts[stop] - obpatient.request_entry_ts[stop],
                              'wait_to_enter': obpatient.wait_to_enter[stop],
                              'wait_to_exit': obpatient.wait_to_exit[stop],
                              'bwaited_to_enter': obpatient.entry_ts[stop] > obpatient.request_entry_ts[stop],
                              'bwaited_to_exit': obpatient.exit_ts[stop] > obpatient.request_exit_ts[stop]}

                obsystem.stops_timestamps_list.append(timestamps)

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

        msg = "{:6}:\t Entries={}, Exits={}, Occ={}, ALOS={:4.2f}". \
            format(self.name, self.num_entries, self.num_exits,
                   self.unit.count, alos)
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

        self.current_stop_num = -1
        self.previous_unit_id = None
        self.current_unit_id = None
        self.next_unit_id = None

        self.route_graph = router.create_route(self.patient_type)
        self.route_length = len(self.route_graph.edges) + 1  # Includes ENTRY and EXIT

        # Since we have fixed route, just initialize full list to hold bed requests
        # The index numbers are stop numbers and so slot 0 is for ENTRY location
        self.bed_requests = [None for _ in range(self.route_length)]
        self.unit_stops = [None for _ in range(self.route_length)]
        self.planned_los = [None for _ in range(self.route_length)]
        self.adjusted_los = [None for _ in range(self.route_length)]
        self.request_entry_ts = [None for _ in range(self.route_length)]
        self.entry_ts = [None for _ in range(self.route_length)]

        self.wait_to_enter = [None for _ in range(self.route_length)]

        self.request_exit_ts = [None for _ in range(self.route_length)]
        self.exit_ts = [None for _ in range(self.route_length)]

        self.wait_to_exit = [None for _ in range(self.route_length)]

        self.system_exit_ts = None

    def __repr__(self):
        return "patientid: {}, patient_type: {}, time: {}". \
            format(self.patient_id, self.patient_type, self.system_arrival_ts)


class OBStaticRouter(object):
    def __init__(self, env, obsystem, locations, routes, rg):
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
        for route_num, route in routes.items():
            route_graph = nx.DiGraph()

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

        logger.debug(
            f"{self.env.now:.4f}:{obpatient.name} current_unit_id {obpatient.current_unit_id}, next_unit_id {next_unit_id}")
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


def process_command_line():
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='obflow_6',
                                     description='Run inpatient OB simulation')

    # Add arguments
    parser.add_argument(
        "config", type=str,
        help="Configuration file containing input parameter arguments and values"
    )

    parser.add_argument("--loglevel", default='WARNING',
                        help="Use valid values for logging package")

    # do the parsing
    args = parser.parse_args()

    # Read inputs from config file
    with open(args.config, 'rt') as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)

    return yaml_config, args.loglevel


def write_stop_log(csv_path, obsystem, egress=False):
    timestamp_df = pd.DataFrame(obsystem.stops_timestamps_list)
    if egress:
        timestamp_df.to_csv(csv_path, index=False)
    else:
        timestamp_df[(timestamp_df['unit'] != 'ENTRY') &
                     (timestamp_df['unit'] != 'EXIT')].to_csv(csv_path, index=False)

    if egress:
        timestamp_df.to_csv(csv_path, index=False)
    else:
        timestamp_df[(timestamp_df['unit'] != 'ENTRY') &
                     (timestamp_df['unit'] != 'EXIT')].to_csv(csv_path, index=False)


def output_header(msg, linelen, scenario, rep_num):
    header = f"\n{msg} (scenario={scenario} rep={rep_num})\n{'-' * linelen}\n"
    return header


def simulate(sim_inputs, rep_num):
    """

    Parameters
    ----------
    sim_inputs : dict whose keys are the simulation input args
    rep_num : int, simulation replication number

    Returns
    -------
    Nothing returned but numerous output files written to ``args_dict[output_path]``

    """

    scenario = sim_inputs['scenario']

    run_settings = sim_inputs['run_settings']
    run_time = run_settings['run_time']
    warmup_time = run_settings['warmup_time']
    global_vars = sim_inputs['global_vars']
    paths = sim_inputs['paths']
    random_number_streams = sim_inputs['random_number_streams']
    locations = sim_inputs['locations']
    routes = sim_inputs['routes']

    stop_log_path = Path(paths['stop_logs']) / f"unit_stop_log_scenario_{scenario}_rep_{rep_num}.csv"
    occ_log_path = Path(paths['occ_logs']) / f"unit_occ_log_scenario_{scenario}_rep_{rep_num}.csv"
    occ_stats_path = Path(paths['occ_stats']) / f"unit_occ_stats_scenario_{scenario}_rep_{rep_num}.csv"

    # Initialize a simulation environment
    env = simpy.Environment()

    # Create an OB System
    obsystem = OBsystem(env, locations, global_vars)

    # Create random number generators
    rg = {}
    for stream, seed in random_number_streams.items():
        rg[stream] = default_rng(seed + rep_num - 1)

    # Create router
    router = OBStaticRouter(env, obsystem, locations, routes, rg['los'])

    # Create patient generator
    obpat_gen = OBPatientGenerator(env, obsystem, router, global_vars['arrival_rate'],
                                   rg['arrivals'], max_arrivals=1000000)

    # Run the simulation replication
    env.run(until=run_time)

    # Compute and display traffic intensities
    header = output_header("Input traffic parameters", 50, scenario, rep_num)
    print(header)

    rho_obs = global_vars['arrival_rate'] * global_vars['mean_los_obs'] / locations[Unit.OBS]['capacity']
    rho_ldr = global_vars['arrival_rate'] * global_vars['mean_los_ldr'] / locations[Unit.LDR]['capacity']
    mean_los_pp = global_vars['mean_los_pp_c'] * global_vars['c_sect_prob'] + \
                  global_vars['mean_los_pp_noc'] * (1 - global_vars['c_sect_prob'])

    rho_pp = global_vars['arrival_rate'] * mean_los_pp / locations[Unit.PP]['capacity']

    print(f"rho_obs: {rho_obs:6.3f}\nrho_ldr: {rho_ldr:6.3f}\nrho_pp: {rho_pp:6.3f}")

    # Patient generator stats
    header = output_header("Patient generator and entry/exit stats", 50, scenario, rep_num)
    print(header)
    print("Num patients generated: {}\n".format(obpat_gen.num_patients_created))

    # Unit stats
    for unit in obsystem.obunits[1:-1]:
        print(unit.basic_stats_msg())

    # System exit stats
    print("\nNum patients exiting system: {}".format(obsystem.obunits[Unit.EXIT].num_exits))
    print("Last exit at: {:.2f}\n".format(obsystem.obunits[Unit.EXIT].last_exit))

    # Create output files
    write_stop_log(stop_log_path, obsystem)

    occ_stats_df = compute_occ_stats(obsystem, run_time,
                                     log_path=occ_log_path,
                                     warmup=warmup_time, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

    occ_stats_df.to_csv(occ_stats_path, index=False)
    header = output_header("Occupancy stats", 50, scenario, rep_num)
    print(header)
    print(occ_stats_df)

    header = output_header("Output logs", 50, scenario, rep_num)
    print(header)
    print(f"Stop log written to {stop_log_path}")
    print(f"Occupancy log written to {occ_log_path}")
    print(f"Occupancy stats written to {occ_stats_path}")


if __name__ == '__main__':

    # Main program

    config, loglevel = process_command_line()

    logging.basicConfig(
        level=loglevel,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=stdout,
    )

    logger = logging.getLogger(__name__)

    num_replications = config['run_settings']['num_replications']
    # run_time = config['run_settings']['run_time']
    # warmup_time = config['run_settings']['warmup_time']
    # paths = config['paths']
    #
    # stop_log_path = Path(paths['stop_logs'])
    # occ_log_path = Path(paths['occ_logs'])
    # occ_stats_path = Path(paths['occ_stats'])
    # output_path = Path(paths['output'])

    # Main simulation replication loop
    for i in range(1, num_replications + 1):
        simulate(config, i)

    # process_obsim_logs(stop_log_path, occ_stats_path, output_path, warmup=warmup_time, run_time=run_time)

    # Consolidate the patient logs and compute summary stats
    # patient_log_stats = process_sim_output(output_dir, scenario)
    # print(f"\nScenario: {scenario}")
    # pd.set_option("display.precision", 3)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 120)
    # print(patient_log_stats['patient_log_rep_stats'])
    # print(patient_log_stats['patient_log_ci'])
