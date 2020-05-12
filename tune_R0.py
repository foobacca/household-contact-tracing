from multiprocessing import Pool
import household_contact_tracing as hct
import pandas as pd
import numpy as np
import numpy.random as npr

# This scripts runs the estimate R0 method for over a range of inputs using parallel processing
# otherwise it is computationallly expensive

# Outputs are saved to Data/calibration_R0_vals.xlsx

#household_hazard_rate = 0.72219

def estimate_R0(hazard_rate_scale):
    """For a given hazard rate scaling, estimates the R_0 value using the model_calibration class

    Arguments:
        hazard_rate_scale {[type]} -- [description]
    """

    model_calibration = hct.model_calibration(haz_rate_scale = hazard_rate_scale,
                                            #household_haz_rate_scale = household_hazard_rate,
                                            contact_tracing_success_prob = 2/3,
                                            contact_trace_delay_par = 1/3,
                                            overdispersion = 0.36,
                                            infection_reporting_prob = 0.7,
                                            contact_trace = True,
                                            reduce_contacts_by = 0)

    return model_calibration.calculate_R0()

x_vals = np.linspace(0.79, 0.83, 100)

if __name__ == '__main__':
    with Pool(14) as p:
        results = p.map(estimate_R0, x_vals)
        results = pd.DataFrame({
            "hazard_rate_scale": x_vals,
            "R0_estimate": results
        })

        results.to_excel("Data/calibration_R0_vals.xlsx")
