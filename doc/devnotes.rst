Useful links
============

Docs
https://simpy.readthedocs.io/en/latest/index.html

Network models
https://www.grotto-networking.com/DiscreteEventPython.html#Intro

One approach to custom Resource
http://simpy.readthedocs.io/en/latest/examples/latency.html


DesMod = New DES package that builds on SimPy
http://desmod.readthedocs.io/en/latest/

Not sure how active. I think I should start with just SimPy to
decide for myself on the metalevel needs in terms of model building,
logging, config files, CLI, etc.

Tidygraph - maybe for representing flow networks visually?
http://www.data-imaginist.com/2017/Introducing-tidygraph/

Vehicle traffic simulation with SUMO
http://www.sumo.dlr.de/userdoc/Sumo_at_a_Glance.html
http://sumo.dlr.de/wiki/Tutorials


Router design
=============


Logging
=======

How best to do trace messages? Is this same use case as "logging"?

In ns-3:

No, tracing is for simulation output and logging for debugging, warnings and errors.

https://www.nsnam.org/docs/release/3.29/manual/html/tracing.html
https://www.nsnam.org/docs/release/3.29/manual/html/data-collection.html

Developing a good tracing system is very important for subsequent
analysis of output and potential animation.

https://docs.python.org/3/library/logging.html

https://bitbucket.org/snippets/benhowes/MKLXy/simpy30-fridge

Software Project Mgt
====================

Semantic versioning seems like a good idea - https://semver.org/
