"""
Scripts to create a specific model and optimizer
from TVB, and to update and run any model
"""

from tvb.analyzers import fmri_balloon
from tvb.contrib.scripts.datatypes import time_series
import numpy as np
from tvb.simulator.models import ModelsEnum
from tvb.simulator import integrators, coupling, monitors


def create_functional_connectivity(conn, sim, res):
    """
    From the output of a TVB model, create a corresponding functional connectivity

    ideally, this function would have different ways to create it, depending on how we
    have created the functional connectivity.

    Use balloon fmri functions? what are those?

    conn: Connectivity of a subject
    sim: simulator object
    res: results of the running simulation
    """

    # Build a TimeSeries Dataype.
    tsr = time_series.TimeSeriesRegion(connectivity=conn,
    data=res, sample_period=sim.monitors[0].period, sample_period_unit='ms')
    # tsr.configure()
    # manually add the time parameter to later compute the balloon, as TVB doesn't do it by default
    # end_time = tsr.start_time + (tsr.data.shape[0] - 1) * tsr.sample_period
    # tsr.time = np.arange(tsr.start_time, end_time + tsr.sample_period, tsr.sample_period)

    # Compute fMRI BOLD activity using Balloon-Windkessel model
    # dt is the integration type step size, in seconds
    # should be equal to the one used in the original fmri
    balloon_analyser = fmri_balloon.BalloonModel(time_series=tsr, dt=0.002) # dt is in seconds
    balloon_data = balloon_analyser.evaluate()
    balloon_data.connectivity = conn
    balloon_data.configure()
    BOLD = balloon_data.data

    return tsr, BOLD

def set_params(obj, params):
    """
    object: any object 
    params: dictionary containing key:value, with key
    being attributes of the object and value being the new value
    return updated object
    """
    for (k,v) in params.items():
        setattr(obj, k, v)
    return obj



def create_tvb_model(model_name, params):
    """
    Create a TVB object model.

    model_name: name of the model
    params: dictionary with the parameters for that model. Should correspond to all the necessary parameters
    for the model
    returns:
    model: the object model wit the set parameters
    """
    # use models_enum to get the class of the model
    model = ModelsEnum[model_name].get_class()

    # set parameters
    model = set_params(model, params)
    return model


def create_tvb_coupling(coupling_name, params):
    """
    Function to create a TVB coupling object, to use in a Simulation.
    coupling_name: the name of the coupling
    params: parameters to configure the coupling object

    returns:
    coupling: the coupling object

    Here there is not Enum, need to select class manually
    """
    if coupling_name == "Linear":
        coupling = coupling.Linear()
    elif coupling_name == "Sigmoidal":
        coupling = coupling.Sigmoidal()
    else:
        raise NotImplementedError(f"{coupling_name} is not implemented!") 

    # set parameters
    coupling = set_params(coupling, params)

    return coupling


def create_tvb_integrator(integrator_name, params):
    """
    Function to define a TVB integrator object, to use in a Simulation.
    integrator_name: name of the integrator
    params: parameters to configure the integrator

    returns:
    integrator: the integrator object

    Here there is not an Enum, need to select class manually
    """
    if integrator_name == "HeunDeterministic":
        integrator = integrators.HeunDeterministic()
    elif integrator_name == "HeunStochastic":
        integrator = integrators.HeunStochastic()
    else:
        raise NotImplementedError(f"{integrator_name} is not implemented!") 

    # set parameters
    integrator = set_params(integrator, params)

    return integrator

def create_tvb_monitor(monitor_name, params):
    """
    Function to define a TVB monitor object, to use in a Simulation.
    monitor_name: name of the monitor
    params: parameters to configure the monitor

    returns:
    monitor: the monitor object

    Only use TemporalAverage and Bold, at the moment
    """
    if monitor_name == "TemporalAverage":
        monitor = monitors.TemporalAverage()
    elif monitor_name == "HeunStochastic":
        monitor = monitors.Bold()
    else:
        raise NotImplementedError(f"{monitor_name} is not implemented!") 

    # set parameters
    monitor = set_params(monitor, params)
    return monitor

def run_model(model, connectivity, coupling, integrator, monitor, simlen, out_file):
    """
    Run a set of specific paramters and models on a single simulation.
    returns the results of the simulation.

    # assumes only a monitor (as we probably will only need one)

    TODO: also, needs to save the results of the simulation to disk if needed.
    """
    sim = simulator.Simulator(
        model=model,
        connectivity=conn,
        coupling=coupling,
        # dt needs to be default or consistent with monitor period (see docs)
        integrator=integrator,
        # monitors=(monitors.Raw(),)
        monitors=(monitor,)
    )
    sim.configure()

    # run the simulation
    (t, y) = sim.run(simulation_length=simlen)

    # save simulation to disk

def fit_model(conn, conn_fc, models_name, params_grid):
    """
    Fit an existing model over a subject for a set of params_grid

    conn: Connectivity object of a subject
    conn_fc: true functional connectivity
    model_name: lists of name of models to try
    params_grid: should be a list, same lengths of model_name, where for each dict,
    the key should be the appropiate parameters and the values, a list of possible params

    # params_grid should not only include the parameters for the model, but also possible parameters for the
    integrator, coupling, etc. So maybe it should be a dict of dicts. We need to see how to implement this

    TODO: MAYBE WE SHOULD DIVIDE THIS BETWEEN FIT_MODEL, for the PARAMETERS, and RUN_MODEL,
    to run a specific model, so that we cna call it independently

    NOTE: this needs to be paralleled
    """
    print('nyi')

    # structures to save the best result and parameters

    # for each model

    # for each combination of params

    # create and run model

    # 

    #