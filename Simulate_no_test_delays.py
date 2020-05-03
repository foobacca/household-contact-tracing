from multiprocessing import Pool
import household_contact_tracing as model
import pandas as pd
import numpy.random as npr

# npr.seed(1)

repeats = 1000
days_to_simulate = 30
starting_infections = 1000

# control parameters
infection_detect_prob = [0.2, 0.4, 0.6, 0.8]
haz_rate_range = [0.816914, 0.819325, 0.809742, 0.806271]

par_range = [
    (0.2, 0.816914),
    (0.4, 0.819325),
    (0.6, 0.809742),
    (0.6, 0.806271)
]

def run_simulation(repeat):

    # Choose a random case detection probability
    random_index = npr.choice(range(4))

    infection_reporting_prob, haz_rate_scale = par_range[random_index]

    contact_tracing_success_prob = npr.uniform(0.7, 0.95)

    contact_trace_delay_par = npr.uniform(1.5, 2.5)

    infection_reporting_prob = npr.uniform(0.2, 0.7)

    reduce_contacts_by = npr.uniform(0.4, 0.9)

    do_2_step = npr.choice([True, False])

    prob_has_trace_app = npr.uniform(0, 0.5)

    simulation = model.household_sim_contact_tracing(haz_rate_scale=haz_rate_scale,
                                                     contact_tracing_success_prob=infection_reporting_prob,
                                                     contact_trace_delay_par=contact_trace_delay_par,
                                                     overdispersion=0.36,
                                                     infection_reporting_prob=infection_reporting_prob,
                                                     contact_trace=True,
                                                     reduce_contacts_by=reduce_contacts_by,
                                                     do_2_step=do_2_step,
                                                     test_before_propagate_tracing=False,
                                                     prob_has_trace_app=prob_has_trace_app,
                                                     starting_infections=starting_infections)

    simulation.run_simulation(days_to_simulate)

    parameters = [
        haz_rate_scale,
        infection_reporting_prob,
        contact_tracing_success_prob,
        contact_trace_delay_par,
        reduce_contacts_by,
        do_2_step,
        prob_has_trace_app
    ]
    return(parameters + simulation.inf_counts)

param_names = [
    "hazard_rate_scale",
    "infection_reporting_prob",
    "contact_tracing_success_prob",
    "contact_trace_delay_par",
    "global_contact_reduction",
    "two_step_tracing",
    "prob_has_trace_app"
]

col_names = param_names + [str(i) for i in range(days_to_simulate)]
col_names_dict = {}
for i in range(len(col_names)):
    col_names_dict.update({i: col_names[i]})

if __name__ == '__main__':
    with Pool(14) as p:
        results = p.map(run_simulation, range(repeats))
        results = pd.DataFrame(results)
        results = results.rename(columns=col_names_dict)
        results.to_excel("simulation_results_no_test_delays.xlsx")
