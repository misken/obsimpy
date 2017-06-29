import simpy

"""
Hello world of SimPy. From tutorial in docs.
"""


# Define an obpatient_flow "process"
def obpatient_flow(env):
    while True:
        print("Entering OBS at {}".format(env.now))
        obs_los = 3
        yield env.timeout(obs_los)

        print("Entering LDR at {}".format(env.now))
        ldr_los = 12
        yield env.timeout(ldr_los)

        print("Entering PP at {}".format(env.now))
        pp_los = 48
        yield env.timeout(pp_los)


# Initialize a simulation environment
env = simpy.Environment()

# Create a process generator and start it and add it to the env
# Calling obpatient(env) creates the generator.
# env.process() starts and adds it to env
env.process(obpatient(env))

# Run the simulation for a while
runtime = 250
env.run(until=runtime)
