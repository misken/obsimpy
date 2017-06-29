import simpy

"""
Class based Hello World of SimPy.
"""


class OBpatient(object):
    def __init__(self, env):
        self.env = env
        # Start the run process everytime an instance is created.
        self.action = env.process(self.run())

        self.obs_los = 3
        self.ldr_los = 12
        self.pp_los = 48

    def run(self):
        while True:
            print("Entering OBS at {}".format(env.now))
            yield self.env.timeout(self.obs_los)

            print("Entering LDR at {}".format(env.now))
            yield self.env.timeout(self.ldr_los)

            print("Entering PP at {}".format(env.now))
            yield self.env.timeout(self.pp_los)


# Initialize a simulation environment
env = simpy.Environment()

# Create an object
obpatient = OBpatient(env)

# Run the simulation for a while
runtime = 250
env.run(until=runtime)
