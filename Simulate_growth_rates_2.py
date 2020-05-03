from multiprocessing import Pool
import household_contact_tracing as hct
import pandas as pd
import numpy as np
import numpy.random as npr
import itertools

# npr.seed(1)

days_to_simulate = 20

param_names = [
    "hazard_rate_scale",
    "infection_reporting_prob"
]

col_names = param_names + [str(i) for i in range(days_to_simulate)]
col_names_dict = {}
for i in range(len(col_names)):
    col_names_dict.update({i: col_names[i]})

par_range = [
    (0.2, 0.816914),
    (0.4, 0.819325),
    (0.6, 0.809742),
    (0.6, 0.806271)
]

def run_simulation(pars):

    infection_reporting_prob, haz_rate = pars

    simulation = hct.household_sim_contact_tracing(haz_rate_scale=haz_rate,
                                                   contact_tracing_success_prob=0,
                                                   contact_trace_delay_par=0,
                                                   overdispersion=0.36,
                                                   infection_reporting_prob=infection_reporting_prob,
                                                   contact_trace=False,
                                                   reduce_contacts_by=0,
                                                   starting_infections=100)

    simulation.run_simulation(days_to_simulate)

    parameters = [
        haz_rate,
        infection_reporting_prob
    ]
    return(parameters + simulation.inf_counts)


if __name__ == '__main__':
    with Pool(14) as p:
        results = p.map(run_simulation, par_range)
        results = pd.DataFrame(results)
        results = results.rename(columns=col_names_dict)
        results.to_excel("Data/simulation_growth_rates_2.xlsx")
