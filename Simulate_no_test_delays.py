from multiprocessing import Pool
import household_contact_tracing as model
import pandas as pd
import numpy.random as npr
import itertools

# npr.seed(1)

repeats = 10000
days_to_simulate = 90

param_names = [
    "R_0",
    "hazard_rate_scale",
    "contact_tracing_success_prob",
    "contact_trace_delay_par",
    "infection_detect_prob",
    "global_contact_reduction",
    "2_step_tracing",
    "prob_has_trace_app"
]

col_names = param_names + [str(i) for i in range(days_to_simulate)]
col_names_dict = {}
for i in range(len(col_names)):
    col_names_dict.update({i: col_names[i]})

R_0_range = [2.6, 2.8, 3.0]
haz_rate_range = [0.829253, 0.816518, 0.803782]


def run_simulation(repeat):

    # Choose a random R_0 value for the simulation:
    R_0_index = npr.choice(range(3))
    R_0 = R_0_range[R_0_index]
    haz_rate_scale = haz_rate_range[R_0_index]

    contact_tracing_success_prob = npr.uniform(0.7, 0.95)

    contact_trace_delay_par = npr.uniform(1.5, 2.5)

    infection_reporting_prob = npr.uniform(0.2, 0.7)

    reduce_contacts_by = npr.uniform(0.4, 0.7)

    do_2_step = npr.choice([True, False])

    prob_has_trace_app = npr.uniform(0, 1)

    simulation = model.household_sim_contact_tracing(haz_rate_scale=haz_rate_scale,
                                                     contact_tracing_success_prob=contact_tracing_success_prob,
                                                     contact_trace_delay_par=contact_trace_delay_par,
                                                     overdispersion=0.36,
                                                     infection_reporting_prob=infection_reporting_prob,
                                                     contact_trace=True,
                                                     reduce_contacts_by=reduce_contacts_by,
                                                     do_2_step=do_2_step,
                                                     test_before_propagate_tracing=False,
                                                     prob_has_trace_app=prob_has_trace_app)

    simulation.run_simulation(days_to_simulate)

    parameters = [
        R_0,
        haz_rate_scale,
        contact_tracing_success_prob,
        contact_trace_delay_par,
        infection_reporting_prob,
        reduce_contacts_by,
        do_2_step,
        prob_has_trace_app
    ]
    return(parameters + simulation.inf_counts)


if __name__ == '__main__':
    with Pool(14) as p:
        results = p.map(run_simulation, range(repeats))
        results = pd.DataFrame(results)
        results = results.rename(columns=col_names_dict)
        results.to_excel("Data/simulation_results_no_test_delays.xlsx")
