from multiprocessing import Pool
import household_contact_tracing as model
import pandas as pd
import numpy.random as npr

# npr.seed(1)

repeats = 1000
days_to_simulate = 60
starting_infections = 1


def run_simulation(repeat):

    haz_rate_scale = 0.818786

    infection_reporting_prob = 0.4

    contact_tracing_success_prob = npr.uniform(0.7, 0.95)

    contact_trace_delay_par = npr.uniform(1.5, 2.5)

 #   reduce_contacts_by = "scenario1_phase1"
    reduce_contacts_by = 0

    do_2_step = npr.choice([True, False])

    prob_has_trace_app = npr.uniform(0, 0.5)

    simulation = model.household_sim_contact_tracing(haz_rate_scale=haz_rate_scale,
                                                     contact_tracing_success_prob=contact_tracing_success_prob,
                                                     contact_trace_delay_par=contact_trace_delay_par,
                                                     overdispersion=0.36,
                                                     infection_reporting_prob=infection_reporting_prob,
                                                     contact_trace=False,
                                                     reduce_contacts_by= 0,
                                                     do_2_step=do_2_step,
                                                     test_before_propagate_tracing=False,
                                                     prob_has_trace_app=prob_has_trace_app,
                                                     starting_infections=starting_infections)

    simulation.run_simulation(days_to_simulate, stop_when_X_infections = True)

    parameters = [
        haz_rate_scale,
        infection_reporting_prob,
        contact_tracing_success_prob,
        contact_trace_delay_par,
        reduce_contacts_by,
        do_2_step,
        prob_has_trace_app
    ]
    
    return(parameters + [simulation.end_reason, simulation.day_extinct] + simulation.inf_counts)  # + simulation.day_ext)

param_names = [
    "hazard_rate_scale",
    "infection_reporting_prob",
    "contact_tracing_success_prob",
    "contact_trace_delay_par",
    "global_contact_reduction",
    "two_step_tracing",
    "prob_has_trace_app"
]

simulation_names = [
    "end_reason",
    "extinction_time"
]


col_names = param_names + simulation_names + [str(i) for i in range(days_to_simulate)]
col_names_dict = {}
for i in range(len(col_names)):
    col_names_dict.update({i: col_names[i]})

if __name__ == '__main__':
    with Pool() as p:
        results = p.map(run_simulation, range(repeats))
        results = pd.DataFrame(results)
        results = results.rename(columns=col_names_dict)
        results.to_excel("Data/simulation_results_ext_times_base_nosocdis_idp40_test.xlsx")


