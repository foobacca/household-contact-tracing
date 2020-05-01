import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import networkx as nx
import scipy as s
import scipy.integrate as si
import math
from matplotlib.lines import Line2D
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or pydot")

# Code for demonstrating contact tracing on networks


# parameters for the generation time distribution
# mean 5, sd = 1.9
gen_shape = 2.85453
gen_scale = 5.61


def weibull_pdf(t):
    out = (gen_shape / gen_scale) * (t / gen_scale)**(gen_shape - 1) * math.exp(-(t / gen_scale)**gen_shape)
    return out


def weibull_hazard(t):
    return (gen_shape / gen_scale) * (t / gen_scale)**(gen_shape - 1)


def weibull_survival(t):
    return math.exp(-(t / gen_scale)**gen_shape)


# Probability of a contact causing infection
def unconditional_hazard_rate(t, survive_forever):
    """
    Borrowed from survival analysis.

    To get the correct generation time distribution, set the probability
    of a contact on day t equal to the generation time distribution's hazard rate on day t

    Since it is not guaranteed that an individual will be infected, we use improper variables and rescale appropriately.
    The R0 scaling parameter controls this, as R0 is closely related to the probability of not being infected
    The relationship does not hold exactly in the household model, hence model tuning is required.

    Notes on the conditional variable stuff https://data.princeton.edu/wws509/notes/c7s1

    Returns:
    The probability that a contact made on day t causes an infection.

    Notes:
    Currently this is using a weibull distribution, as an example.
    """
    unconditional_pdf = (1 - survive_forever) * weibull_pdf(t)
    unconditional_survival = (1 - survive_forever) * weibull_survival(t) + survive_forever
    return unconditional_pdf / unconditional_survival


def current_prob_infection(t, survive_forever):
    """Integrates over the unconditional hazard rate to get the probability of a contact causing infection on day t.

    survive_forever controls the probability that an infection never occurs, and is important to set R0.

    Arguments:
        t {int} -- current day
        survive_forever {float} -- rescales the hazard rate so that it is possible to not be infected
    """
    return si.quad(lambda t: unconditional_hazard_rate(t, survive_forever), t, t+1)[0]


def negbin_pdf(x, m, a):
    """
    We need to draw values from an overdispersed negative binomial distribution, with non-integer inputs. Had to 
    generate the numbers myself in order to do this.
    This is the generalized negbin used in glm models I think.

    m = mean
    a = overdispertion
    """
    A = math.gamma(x + 1 / a) / (math.gamma(x + 1) * math.gamma(1 / a))
    B = (1 / (1 + a * m))**(1 / a)
    C = (a * m / (1 + a * m))**x
    return A * B * C


def compute_negbin_cdf(mean, overdispersion, length_out):
    """
    Computes the overdispersed negative binomial cdf, which we use to generate random numbers by generating uniform(0,1)
    rv's.
    """
    pdf = [negbin_pdf(i, mean, overdispersion) for i in range(length_out)]
    cdf = [sum(pdf[:i]) for i in range(length_out)]
    return cdf


# Precomputing the cdf's for generating the overdispersed contact data, saves a lot of time later


class household_sim_contact_tracing:
    # We assign each node a recovery period of 14 days, after 14 days the probability of causing a new infections is 0,
    # due to the generation time distribution
    effective_infectious_period = 21

    # Working out the parameters of the incubation period
    ip_mean = 4.83
    ip_var = 2.78**2
    ip_scale = ip_var / ip_mean
    ip_shape = ip_mean ** 2 / ip_var

    # Visual Parameters:
    contact_traced_edge_colour_within_house = "blue"
    contact_traced_edge_between_house = "magenta"
    default_edge_colour = "black"
    failed_contact_tracing = "red"
    app_traced_edge = "green"

    def __init__(self,
                 haz_rate_scale,
                 contact_tracing_success_prob,
                 contact_trace_delay_par,
                 overdispersion,
                 infection_reporting_prob,
                 contact_trace,
                 do_2_step=False,
                 reduce_contacts_by=1,
                 prob_has_trace_app=0,
                 test_delay_mean=1.52,
                 test_before_propagate_tracing=True,
                 starting_infections=1,
                 hh_prob_propensity_to_leave_isolation=0):
        """Initializes parameters and distributions for performing a simulation of contact tracing.
        The epidemic is modelled as a branching process, with nodes assigned to households.

        Arguments:
            proportion_of_within_house_contacts {[type]} -- [description]
            haz_rate_scale {[type]} -- controls the R_0 by rescaling the hazard rate function
            contact_tracing_success_prob {[type]} -- [description]
            prob_of_successful_contact_trace_today {[type]} -- [description]
        """
        # Probability of each household size
        house_size_probs = [0.294591195, 0.345336927, 0.154070081, 0.139478886, 0.045067385, 0.021455526]

        # Size biased distribution of households (choose a node, what is the prob they are in a house size 6, this is 
        # biased by the size of the house)
        size_biased_distribution = [(i + 1) * house_size_probs[i] for i in range(6)]
        total = sum(size_biased_distribution)
        self.size_biased_distribution = [prob / total for prob in size_biased_distribution]

        # The mean number of contacts made by each household
        means = [8.87, 10.65, 12.87, 15.84, 16.47, 17.69]
        self.proportion_of_within_house_contacts = [0.3263158, 0.2801083, 0.3002421, 0.3545259, 0.3683640, 0.4122288]

        # Precomputing the cdf's for generating the overdispersed contact data
        self.cdf_dict = {
            1: compute_negbin_cdf(means[0], overdispersion, 100),
            2: compute_negbin_cdf(means[1], overdispersion, 100),
            3: compute_negbin_cdf(means[2], overdispersion, 100),
            4: compute_negbin_cdf(means[3], overdispersion, 100),
            5: compute_negbin_cdf(means[4], overdispersion, 100),
            6: compute_negbin_cdf(means[5], overdispersion, 100)
        }

        # Parameter Inputs:
        self.haz_rate_scale = haz_rate_scale
        self.contact_tracing_success_prob = contact_tracing_success_prob
        self.contact_trace_delay_par = contact_trace_delay_par
        self.overdispersion = overdispersion
        self.infection_reporting_prob = infection_reporting_prob
        self.contact_trace = contact_trace
        self.prob_has_trace_app = prob_has_trace_app
        self.reduce_contacts_by = reduce_contacts_by
        self.do_2_step = do_2_step
        self.test_before_propagate_tracing = test_before_propagate_tracing
        self.test_delay_mean = test_delay_mean
        self.starting_infections = starting_infections
        self.hh_prob_propensity_to_leave_isolation = hh_prob_propensity_to_leave_isolation
        if do_2_step:
            self.max_tracing_index = 2
        else:
            self.max_tracing_index = 1

        # Calls the simulation reset function, which creates all the required dictionaries
        self.reset_simulation()

    def contact_trace_delay(self, app_traced_edge):
        if app_traced_edge:
            return 0
        else:
            return npr.poisson(self.contact_trace_delay_par)

    def incubation_period(self):
        return round(npr.gamma(
            shape=self.ip_shape,
            scale=self.ip_scale))

    def testing_delay(self):
        if self.test_before_propagate_tracing is False:
            return 0
        else:
            return round(npr.gamma(
                shape=self.test_delay_mean**2 / 1.11**2,
                scale=1.11**2 / self.test_delay_mean))

    def reporting_delay(self):
        return round(npr.gamma(
            shape=2.62**2/2.38**2,
            scale=2.38**2/2.62))

    def hh_propensity_to_leave_isolation(self):
        if npr.binomial(1, self.hh_prob_propensity_to_leave_isolation) == 1:
            return True
        else:
            return False

    def contacts_made_today(self, household_size):
        """Generates the number of contacts made today by a node, given the house size of the node. Uses an 
        overdispersed negative binomial distribution.

        Arguments:
            house_size {int} -- size of the nodes household
        """
        random = npr.uniform()
        cdf = self.cdf_dict[household_size]
        obs = sum([int(cdf[i] < random) for i in range(100)])
        return obs

    def size_of_household(self):
        """Generates a random household size

        Returns:
        household_size {int}
        """
        return npr.choice([1, 2, 3, 4, 5, 6], p=self.size_biased_distribution)

    def has_contact_tracing_app(self):
        return npr.binomial(1, self.prob_has_trace_app) == 1

    def count_non_recovered_nodes(self):
        """Returns the number of nodes not in the recovered state.
        Returns:
            [int] -- Number of non-recovered nodes.
        """
        return len([node for node in self.G.nodes() if self.G.nodes[node]["recovered"]])

    def new_infection(self, node_count, generation, household, serial_interval=None):
        """
        Adds a new infection to the graph along with the following attributes:
        t - when they were infected
        offspring - how many offspring they produce

        Inputs::
        G - the network object
        time - the time when the new infection happens
        node_count - how many nodes are currently in the network
        """
        # Symptom onset time
        symptom_onset_time = self.time + self.incubation_period()
        # When a node reports it's infection
        if npr.binomial(1, self.infection_reporting_prob) == 1:
            will_report = True
            time_of_reporting = symptom_onset_time + self.reporting_delay()
        else:
            will_report = False
            time_of_reporting = float('Inf')

        # We assign each node a recovery period of 21 days, after 21 days the probability of causing a new infections is
        # 0, due to the generation time distribution
        recovery_time = self.time + 21

        # Give the node the required attributes
        self.G.add_node(node_count)
        new_node_dict = {
            "time_infected": self.time,
            "generation": generation,
            "household": household,
            "contact_traced": False,
            "isolated": self.house_dict[household]["isolated"],
            "symptom_onset": symptom_onset_time,
            "outside_house_contacts_made": 0,
            "had_contacts_traced": False,
            "spread_to": [],
            "serial_interval": serial_interval,
            "recovered": False,
            "recovery_time": recovery_time,
            "will_report_infection": will_report,
            "reporting_time": time_of_reporting,
            "has_trace_app": self.has_contact_tracing_app(),
            "testing_delay": self.testing_delay()
        }

        self.G.nodes[node_count].update(new_node_dict)

        # Updates to the household dictionary
        # Each house now stores a the ID's of which nodes are stored inside the house, so that quarantining can be done at the household level
        self.house_dict[household]['nodes'].append(node_count)

    def new_household(self, new_household_number, generation, infected_by, infected_by_node):
        """Adds a new household to the household dictionary
        
        Arguments:
            new_household_number {int} -- The house id
            generation {int} -- The household generation of this household
            infected_by {int} -- Which household spread the infection to this household
            infected_by_node {int} -- Which node spread the infection to this household
        """
        house_size = self.size_of_household()

        propensity = self.hh_propensity_to_leave_isolation()

        self.house_dict.update({new_household_number:
                                { 
                                    "size": house_size,                  # Size of the household
                                    "time": self.time,                   # The time at which the infection entered the household
                                    "susceptibles": house_size - 1,      # How many susceptibles remain in the household
                                    "isolated": False,                   # Has the household been isolated, so there can be no more infections from this household
                                    "isolated_time": float('inf'),       # When the house was isolated
                                    "propensity_to_leave_isolation": propensity,
                                    "contact_traced": False,             # If the house has been contact traced, it is isolated as soon as anyone in the house shows symptoms
                                    "time_until_contact_traced": float('inf'),# The time until quarantine, calculated from contact tracing processes on connected households
                                    "contact_traced_households": [],     # The list of households contact traced from this one
                                    "being_contact_traced_from": None,   # If the house if being contact traced, this is the house_id of the first house that will get there
                                    "propagated_contact_tracing": False,  # The house has not yet propagated contact tracing
                                    "time_propagated_tracing": None,     # Time household propagated contact tracing
                                    "contact_tracing_index": 0,          # The house is which step of the contact tracing process
                                    "generation": generation,            # Which generation of households it belongs to
                                    "infected_by": infected_by,          # Which house infected the household
                                    "spread_to": [],                     # Which households were infected by this household
                                    "nodes": [],                         # The ID of currently infected nodes in the household
                                    "infected_by_node": infected_by_node,# Which node infected the household
                                    "within_house_edges": [],            # Which edges are contained within the household
                                    "had_contacts_traced": False         # Have the nodes inside the household had their contacts traced?
                                }
                            })

    def get_edge_between_household(self, house_1, house_2):
        for node_1 in self.house_dict[house_1]["nodes"]:
            for node_2 in self.house_dict[house_2]["nodes"]:
                if self.G.has_edge(node_1, node_2):
                    return (node_1, node_2)

    def is_edge_app_traced(self, edge):
        """Returns whether both ends of an edge have the app, and the app does the tracing.
        """
        node_1_app = self.G.nodes[edge[0]]["has_trace_app"]
        node_2_app = self.G.nodes[edge[1]]["has_trace_app"]
        return (node_2_app and node_1_app)

    @property
    def active_infections(self):
        return [
            node for node in self.G.nodes()
            if (self.G.nodes[node]["reporting_time"] >= self.time and
                self.G.nodes[node]["recovered"] is False)
        ]

    def increment_infection(self):
        """
        Creates a new days worth of infections
        """

        for node in self.active_infections:

            # Extracting useful parameters from the node
            node_household = self.G.nodes()[node]["household"]
            days_since_infected = self.time - self.G.nodes()[node]["time_infected"]
            household_size = self.house_dict[node_household]["size"]

            # The number of contacts made that will could spread the infection, if the other person is susceptible
            contacts_made = self.contacts_made_today(household_size)

            # How many of the contacts are within the household
            within_household_contacts = npr.binomial(contacts_made, self.proportion_of_within_house_contacts[household_size-1])

            # Each contact is with a unique individual, so it is not possible to have more than h-1 contacts within household
            within_household_contacts = min(household_size - 1, within_household_contacts)

            # Work out how many contacts were with other households
            # If social distancing is in play, global contacts are reduced by
            if self.G.nodes[node]["isolated"] is False:
                outside_household_contacts = round((1-self.reduce_contacts_by)*(contacts_made - within_household_contacts))
            else:
                outside_household_contacts = 0

            # Within household, how many of the infections would cause new infections
            # These contacts may be made with someone who is already infected, and so they will again be thinned
            within_household_potential_new_infections = npr.binomial(within_household_contacts, current_prob_infection(days_since_infected, self.haz_rate_scale))

            for _ in range(within_household_potential_new_infections):
                # A further thinning has to happen since each attempt may choose an already infected person
                # That is to say, if everyone in your house is infected, you have 0 chance to infect a new person in your house

                # Get the number of susceptibles for their current household
                susceptibles = self.house_dict[node_household]["susceptibles"]

                # A one represents a susceptibles node in the household
                # A 0 represents an infected member of the household
                # We choose a random subset of this vector of length within_household_potential_new_infections to determine infections
                household_composition = [1]*susceptibles + [0]*(household_size - 1 - susceptibles)
                within_household_new_infections = sum(npr.choice(household_composition, within_household_potential_new_infections, replace=False))

                # If the within household infection is successful:
                for _ in range(within_household_new_infections):
                    self.new_within_household_infection(
                        infecting_node=node,
                        serial_interval=days_since_infected
                    )

            # Update how many contacts the node made
            self.G.nodes()[node]["outside_house_contacts_made"] += outside_household_contacts

            # How many outside household contacts cause new infections
            outside_household_new_infections = npr.binomial(outside_household_contacts, current_prob_infection(days_since_infected, self.haz_rate_scale))

            for _ in range(outside_household_new_infections):
                self.new_outside_household_infection(
                    infecting_node=node,
                    serial_interval=days_since_infected)

    def new_within_household_infection(self, infecting_node, serial_interval):
        # Add a new node to the network, it will be a member of the same household that the node that infected it was
        node_count = nx.number_of_nodes(self.G) + 1

        # We record which node caused this infection
        self.G.nodes()[infecting_node]["spread_to"].append(node_count)

        infecting_node_household = self.G.nodes[infecting_node]["household"]

        # Adds the new infection to the network
        self.new_infection(node_count=node_count,
                           generation=self.G.nodes()[infecting_node]["generation"] + 1,
                           household=infecting_node_household,
                           serial_interval=serial_interval)

        # Add the edge to the graph and give it the default colour
        self.G.add_edge(infecting_node, node_count)
        self.G.edges[infecting_node, node_count].update({"colour": self.default_edge_colour})

        # Decrease the number of susceptibles in that house by 1
        self.house_dict[infecting_node_household]['susceptibles'] -= 1

        # We record which edges are within this household for visualisation later on
        self.house_dict[infecting_node_household]["within_house_edges"].append((infecting_node, node_count))

    def new_outside_household_infection(self, infecting_node, serial_interval):
        # We assume all new outside household infections are in a new household
        # i.e: You do not infect 2 people in a new household
        # you do not spread the infection to a household that already has an infection
        self.house_count += 1
        node_count = nx.number_of_nodes(self.G) + 1
        infecting_household = self.G.nodes[infecting_node]["household"]

        # We record which node caused this infection
        self.G.nodes()[infecting_node]["spread_to"].append(node_count)

        # We record which house spread to which other house
        self.house_dict[infecting_household]["spread_to"].append(self.house_count)

        # Create a new household, since the infection was outside the household
        self.new_household(new_household_number=self.house_count,
                           generation=self.house_dict[infecting_household]["generation"] + 1,
                           infected_by=self.G.nodes()[infecting_node]["household"],
                           infected_by_node=infecting_node)

        # add a new infection in the house just created
        self.new_infection(node_count=node_count,
                           generation=self.G.nodes()[infecting_node]["generation"] + 1,
                           household=self.house_count,
                           serial_interval=serial_interval)

        # Add the edge to the graph and give it the default colour
        self.G.add_edge(infecting_node, node_count)
        self.G.edges[infecting_node, node_count].update({"colour": "black"})

    def increment_contact_tracing(self):
        """
        Performs a days worth of contact tracing by:
        * Looking for nodes that have been admitted to hospital. Once a node is admitted to hospital, it's house is isolated
        * Looks for houses where the contact tracing delay is over and moves them to the contact traced state
        * Looks for houses in the contact traced state, and checks them for symptoms. If any of them have symptoms, the house is isolated

        The isolation function also assigns contact tracing times to any houses that had contact with that household

        For each node that is contact traced
        """

        # Update the contact traced status for all households that have had the contact tracing process get there
        [
            self.contact_trace_household(house)
            for house in self.house_dict
            if (
                self.house_dict[house]["time_until_contact_traced"] <= self.time and
                self.house_dict[house]["contact_traced"] is False)
        ]

        # Isolate all non isolated households where the infection has been reported
        [
            self.isolate_household(self.G.nodes()[node]["household"])
            for node in self.G.nodes()
            if (self.G.nodes()[node]["reporting_time"] <= self.time and
                self.G.nodes()[node]["isolated"] is False)
        ]

        # Isolate all households under observation that now display symptoms
        # TODO can this be removed?
        [
            self.isolate_household(self.G.nodes[node]["household"])
            for node in self.G.nodes()
            if (self.G.nodes[node]["symptom_onset"] <= self.time and
                self.G.nodes[node]["contact_traced"] is True and
                self.G.nodes[node]["isolated"] is False)
        ]

        # Look for houses that need to propagate the contact tracing because their test result has come back
        # Necessary conditions: household isolated, symptom onset + testing delay = time

        # Propagate the contact tracing for all households that self-reported and have had their test results come back
        [
            self.propagate_contact_tracing(self.G.nodes[node]["household"])
            for node in self.G.nodes() 
            if (self.G.nodes[node]["reporting_time"] + self.G.nodes[node]["testing_delay"] == self.time and
                self.house_dict[self.G.nodes[node]["household"]]["propagated_contact_tracing"] is False)
        ]

        # Propagate the contact tracing for all households that are isolated due to exposure, have developed symptoms and had a test come back
        [
            self.propagate_contact_tracing(self.G.nodes[node]["household"])
            for node in self.G.nodes()
            if (self.G.nodes[node]["symptom_onset"] <= self.time and
                self.house_dict[self.G.nodes[node]["household"]]["propagated_contact_tracing"] is False and
                self.house_dict[self.G.nodes[node]["household"]]["isolated_time"] + self.G.nodes[node]["testing_delay"] <= self.time)
        ]

        # Update the contact tracing index of households
        # That is, re-evaluate how far away they are from a known infected case (traced + symptom_onset + testing_delay)
        self.update_contact_tracing_index()

        if self.do_2_step:

            # Propagate the contact tracing from any households with a contact tracing index of 1
            [
                self.propagate_contact_tracing(household)
                for household in self.house_dict if (
                    self.house_dict[household]["contact_tracing_index"] == 1 and
                    self.house_dict[household]["propagated_contact_tracing"] is False and
                    self.house_dict[household]["isolated"] is True
                )
            ]

        
        # # The following chunk of code is to record counts of how many contacts must be traced, used for evaluating when capacity is reached
        # # Get all the cases that have reported their infection today
        # newly_reported = [node for node in self.G.nodes() if (
        #     self.G.nodes()[node]["had_contacts_traced"] == False 
        #     and self.G.nodes()[node]["reporting_time"] == self.time)
        #     ]
        
        # # For nodes who have just onset symptoms, but their household has been contact traced, now trace their contacts
        # new_symptomatic = [node for node in self.G.nodes() if (self.G.nodes()[node]["had_contacts_traced"] == False 
        #                                                         and self.G.nodes()[node]["symptom_onset"] == self.time 
        #                                                         and self.G.nodes()[node]["contact_traced"] == True)]

        # households_to_be_traced = [self.G.nodes[node]["household"] for node in (newly_reported + new_symptomatic)]
        
        # # de-duplicating
        # households_to_be_traced = list(set(households_to_be_traced))

        # contacts_to_be_traced = 0

        # # We work out how many nodes must now have their contacts traced:
        # for house in households_to_be_traced:
            
        #     # All contacts in house are being traced when the 
        #     household_size = self.house_dict[house]["size"]

        #     for node in range(household_size):
        #         for _ in range(14):
        #             contacts_made = self.contacts_made_today(household_size)
                
        #             # How many of the contacts are within the household
        #             within_household_contacts = npr.binomial(contacts_made, self.proportion_of_within_house_contacts[household_size-1])
                    
        #             # Each contact is with a unique individual, so it is not possible to have more than h-1 contacts within household
        #             within_household_contacts = min(household_size - 1, within_household_contacts)
                    
        #             # Work out how many contacts were with other households
        #             # If social distancing is in play, global contacts are reduced by
        #             outside_household_contacts = round((1-self.reduce_contacts_by)*(contacts_made - within_household_contacts))

        #             contacts_to_be_traced += outside_household_contacts

        # # update the total contacts to be traced
        # total_contacts_to_be_traced = self.contact_tracing_dict["contacts_to_be_traced"] + contacts_to_be_traced
        # self.contact_tracing_dict.update({"contacts_to_be_traced": total_contacts_to_be_traced})
            
        # # Some contacts will never be traced, these are assumed to immediately be removed
        # possible_to_trace = npr.binomial(contacts_to_be_traced, self.contact_tracing_success_prob)
        # total_possible_to_trace = self.contact_tracing_dict["possible_to_trace_contacts"] + possible_to_trace
        # self.contact_tracing_dict.update({"possible_to_trace_contacts": total_possible_to_trace})
            
        # # Work out how many were successfully traced today (this is where the geometric assumption becomes very useful)
        # successful_contact_traces = npr.binomial(self.contact_tracing_dict["possible_to_trace_contacts"], 0.7)
        # total_possible_to_trace = self.contact_tracing_dict["possible_to_trace_contacts"] - successful_contact_traces
        # self.contact_tracing_dict.update({"possible_to_trace_contacts": total_possible_to_trace})
        
        # # Add another entry to the time series
        # trace_log = self.contact_tracing_dict["total_traced_each_day"]
        # trace_log.append(successful_contact_traces)
        # self.contact_tracing_dict.update({"total_traced_each_day": trace_log})
        
        # # Record how many active surveillances there are
        # if self.time > 14:
        #     currently_being_surveilled = sum(self.contact_tracing_dict["total_traced_each_day"][-14:])
        # else:
        #     currently_being_surveilled = sum(self.contact_tracing_dict["total_traced_each_day"])
        # self.contact_tracing_dict.update({"currently_being_surveilled": currently_being_surveilled})
        
        # # Daily active surveillances
        # daily_active_surveillances = self.contact_tracing_dict["daily_active_surveillances"]
        # daily_active_surveillances.append(currently_being_surveilled)
        # self.contact_tracing_dict.update({"daily_active_surveillances": daily_active_surveillances})

    def contact_trace_household(self, household_number):
        """
        When a house is contact traced, we need to place all the nodes under surveillance.

        If any of the nodes are symptomatic, we need to isolate the household.
        """

        # Update the house to the contact traced status
        self.house_dict[household_number].update({"contact_traced": True})

        # Update the nodes to the contact traced status
        [self.G.nodes()[node].update({"contact_traced": True}) for node in self.house_dict[household_number]["nodes"]]

        # Colour the edges within household
        [self.G.edges[edge[0], edge[1]].update({"colour": self.contact_traced_edge_colour_within_house}) for edge in self.house_dict[household_number]["within_house_edges"]]

        # If there are any nodes in the house that are symptomatic, isolate the house:
        symptomatic_nodes = [node for node in self.house_dict[household_number]["nodes"] if self.G.nodes()[node]["symptom_onset"] <= self.time]
        if symptomatic_nodes != []:
            self.isolate_household(household_number)
        else:
            self.isolate_household(household_number)

    def perform_recoveries(self):
        """
        Loops over all nodes in the branching process and determins recoveries.

        time - The current time of the process, if a nodes recovery time equals the current time, then it is set to the recovered state
        """
        [self.G.nodes()[node].update({"recovered": True}) for node in self.G.nodes() if self.G.nodes[node]["recovery_time"] == self.time]

    def colour_node_edges_between_houses(self, house_to, house_from, new_colour):

        # Annoying bit of logic to find the edge and colour it
        for node_1 in self.house_dict[house_to]["nodes"]:
            for node_2 in self.house_dict[house_from]["nodes"]:
                if self.G.has_edge(node_1, node_2):
                    self.G.edges[node_1, node_2].update({"colour": new_colour})

    def attempt_contact_trace_of_household(self, house_to, house_from, contact_trace_delay=0):
        # Decide if the edge was traced by the app
        app_traced = self.is_edge_app_traced(self.get_edge_between_household(house_from, house_to))

        # Get the success probability
        if app_traced:
            success_prob = 1
        else:
            success_prob = self.contact_tracing_success_prob

        # is the trace successful
        if (npr.binomial(1, success_prob) == 1):
            
            # Update the list of traced households from this one
            self.house_dict[house_from]["contact_traced_households"].append(house_to)

            # Assign the household a contact tracing index, 1 more than it's parent tracer
            self.house_dict[house_to].update({"contact_tracing_index": self.house_dict[house_from]["contact_tracing_index"] + 1})

            # work out the time delay
            contact_trace_delay = contact_trace_delay + self.contact_trace_delay(app_traced)
            proposed_time_until_contact_trace = self.time + contact_trace_delay

            # Get the current time until contact trace, and compare against the proposed time until contact trace
            # Note this starts as infinity
            time_until_contact_trace = self.house_dict[house_to]["time_until_contact_traced"]

            # If the new proposed time is quicker, change the route
            if proposed_time_until_contact_trace < time_until_contact_trace:
                self.house_dict[house_to].update({"time_until_contact_traced": proposed_time_until_contact_trace})
                self.house_dict[house_to].update({"being_contact_traced_from": house_from})

            # should this only be done if the route is used (above block)
            # if trace_neighbours:
            #     for node_from in self.house_dict[house_to]["nodes"]:
            #         for node_to in nx.neighbors(self.G, node_from):

            #             node_to_household = self.G.nodes[node_to]["household"]

            #             if self.house_dict[node_to_household]["contact_traced"] == False:
            #                 self.attempt_contact_trace_of_household(node_to_household, house_to, contact_trace_delay, trace_neighbours=False)
            
            # Edge colouring
            if app_traced:
                self.colour_node_edges_between_houses(house_to, house_from, self.app_traced_edge)
            else:
                self.colour_node_edges_between_houses(house_to, house_from, self.contact_traced_edge_between_house)
        else:
            self.colour_node_edges_between_houses(house_to, house_from, self.failed_contact_tracing)

    def isolate_household(self, household_number):
        """
        Isolates a house so that all infectives in that household may no longer infect others.

        If the house is being surveillance due to a successful contact trace, and not due to reporting symptoms,
        update the edge colour to display this.

        For households that were connected to this household, they are assigned a time until contact traced

        When a house has been contact traced, all nodes in the house are under surveillance for symptoms. When a node becomes symptomatic, the house moves to isolation status.
        """

        # The house moves to isolated status
        self.house_dict[household_number].update({"isolated": True})
        #self.house_dict[household_number].update({"contact_traced": True})
        self.house_dict[household_number].update({"isolated_time": self.time})
        
        # Update every node in the house to the isolated status
        infectives_in_house = self.house_dict[household_number]["nodes"]
        for node in infectives_in_house:
            self.G.nodes()[node].update({"isolated": True})
            self.G.nodes()[node].update({"contact_traced": True})

        # Which house started the contact trace that led to this house being isolated, if there is one
        # A household may be being isolated because someone in the household self reported symptoms
        # Hence sometimes there is a None value for House which contact traced
        house_which_contact_traced = self.house_dict[household_number]["being_contact_traced_from"]

        # Initially the edge is assigned the contact tracing colour, may be updated if the contact tracing does not succeed
        if house_which_contact_traced:
            if self.is_edge_app_traced(self.get_edge_between_household(household_number, house_which_contact_traced)):
                self.colour_node_edges_between_houses(household_number, house_which_contact_traced, self.app_traced_edge)
            else:
                self.colour_node_edges_between_houses(household_number, house_which_contact_traced, self.contact_traced_edge_between_house)

        # We update the colour of every edge so that we can tell which household have been contact traced when we visualise
        [
            self.G.edges[edge[0], edge[1]].update({"colour": self.contact_traced_edge_colour_within_house})
            for edge in self.house_dict[household_number]["within_house_edges"]
        ]

    def propagate_contact_tracing(self, household_number):
        """
        To be called after a node in a household either reports their symptoms, and gets tested, when a household that is under surveillance develops symptoms + gets tested.
        """

        # update the propagation data
        self.house_dict[household_number].update({"propagated_contact_tracing": True})
        self.house_dict[household_number].update({"time_propagated_tracing": self.time})

        # Contact tracing attempted for the household that infected the household currently propagating the infection
        infected_by = self.house_dict[household_number]["infected_by"]

        # If infected by = None, then it is the origin node, a special case
        if infected_by is None:
            is_infector_traced = True
        else:
            is_infector_traced = self.house_dict[infected_by]["isolated"]

        if is_infector_traced is False:
            self.attempt_contact_trace_of_household(infected_by, household_number)

        # Contact tracing for the households infected by the household currently traced
        child_households = self.house_dict[household_number]["spread_to"]

        child_households_not_traced = [house for house in child_households if self.house_dict[house]["isolated"] == False]

        for house in child_households_not_traced:
            self.attempt_contact_trace_of_household(house, household_number)

    def update_contact_tracing_index(self):
        for household in self.house_dict:

            # loop over households with non-zero indexes, those that have been contact traced but with 
            if self.house_dict[household]["contact_tracing_index"] != 0:

                for node in self.house_dict[household]["nodes"]:
                    
                    # Necessary conditions for an index 1 household to propagate tracing:
                    # The node must have onset of symptoms
                    # The node households must be isolated
                    # The testing delay must be passed
                    # The testing delay starts when the house have been isolated and symptoms have onset
                    critical_time = max(self.G.nodes[node]["symptom_onset"], self.house_dict[household]["isolated_time"])

                    if (critical_time + self.G.nodes[node]["testing_delay"] <= self.time):

                        self.house_dict[household].update({"contact_tracing_index": 0})

                        for index_1_hh in self.house_dict[household]["contact_traced_households"]:
                            if self.house_dict[index_1_hh]["contact_tracing_index"] == 2:
                                self.house_dict[index_1_hh].update({"contact_tracing_index": 1})

    def simulate_one_day(self):
        """Simulates one day of the epidemic and contact tracing.

        Useful for bug testing and visualisation.
        """
        self.increment_infection()
        if self.contact_trace is True:
            for _ in range(5):
                self.increment_contact_tracing()
        self.perform_recoveries()
        self.time += 1

    def reset_simulation(self):
        """
        Returns the simulation to it's initially specified values
        """

        self.time = 0

        # Stores information about the households.
        self.house_dict = {}

        # Stores information about the contact tracing that has occurred.
        self.contact_tracing_dict = {
            "contacts_to_be_traced": 0,         # connections made by nodes that are contact traced and symptomatic
            "possible_to_trace_contacts": 0,    # contacts that are possible to trace assuming a failure rate, not all connections will be traceable
            "total_traced_each_day": [0],       # A list recording the the number of contacts added to the system each day
            "daily_active_surveillances": [],   # A list recording how many surveillances were happening each day
            "currently_being_surveilled": 0,    # Ongoing surveillances
            "day_800_cases_traced": None        # On which day was 800 cases reached
        }

        # Create the empty graph
        self.G = nx.Graph()

        # Create first household
        self.house_count = 0

        # Initial values
        node_count = 1
        generation = 0

        # Create the starting infectives
        for _ in range(self.starting_infections):
            self.house_count += 1
            node_count = nx.number_of_nodes(self.G) + 1
            self.new_household(self.house_count, 1, None, None)
            self.new_infection(node_count, generation, self.house_count)

    def run_simulation_hitting_times(self, time_out):

        # Return the simulation to it's initial starting state
        self.reset_simulation()

        # Get the number of current nodes in the network
        node_count = nx.number_of_nodes(self.G)

        # For recording the number of cases over time
        total_cases = []

        # Setting up parameters for this run of the experiment
        self.time_800 = None    # Time when we hit 800 under surveillance
        self.time_8000 = None   # Time when we hit 8000 under surveillance
        self.hit_800 = False    # flag used for the code that records the first time we hit 800 under surveillance
        self.hit_8000 = False   # same but for 8000
        self.died_out = False   # flag for whether the epidemic has died out
        self.timed_out = False  # flag for whether the simulation reached it's time limit without another stop condition being met

        # While loop ends when there are no non-isolated infections
        currently_infecting = len([node for node in self.G.nodes() if self.G.nodes[node]["recovered"] is False])

        while (currently_infecting != 0 and self.hit_8000 is False and self.timed_out is False):

            # This chunk of code executes a days worth on infections and contact tracings
            node_count = nx.number_of_nodes(self.G)
            self.simulate_one_day()

            self.house_count = len(self.house_dict)
            total_cases.append(node_count)

            # While loop ends when there are no non-isolated infections
            currently_infecting = len([node for node in self.G.nodes() if self.G.nodes()[node]["isolated"] is False])

            # Records the first time we hit 800 under surveillance
            if (self.contact_tracing_dict["currently_being_surveilled"] > 800 and self.hit_800 is False):
                self.time_800 = self.time
                self.hit_800 = True

            # Records the first time we hit 8000 surveilled
            if (self.contact_tracing_dict["currently_being_surveilled"] > 8000 and self.hit_8000 is False):
                self.time_8000 = self.time
                self.hit_8000 = True

            if currently_infecting == 0:
                self.died_out = True

            if self.time == time_out:
                self.timed_out = True

        # Infection Count output
        self.inf_counts = total_cases

    def run_simulation(self, time_out):

        # Create all the required dictionaries and reset parameters
        self.reset_simulation()

        # For recording the number of cases over time
        self.total_cases = []

        # Initial values
        self.timed_out = False
        self.extinct = False

        # While loop ends when there are no non-isolated infections
        currently_infecting = len([node for node in self.G.nodes() if self.G.nodes[node]["recovered"] is False])

        while (currently_infecting != 0 and self.timed_out is False):

            # This chunk of code executes a days worth on infections and contact tracings
            node_count = nx.number_of_nodes(self.G)
            self.simulate_one_day()

            self.house_count = len(self.house_dict)
            self.total_cases.append(node_count)

            # While loop ends when there are no non-isolated infections
            currently_infecting = len([node for node in self.G.nodes() if self.G.nodes[node]["recovered"] is False])

            if currently_infecting == 0:
                self.died_out = True

            if self.time == time_out:
                self.timed_out = True

        # Infection Count output
        self.inf_counts = self.total_cases

    def onset_to_isolation_times(self, include_self_reports = True):
        if include_self_reports:
            return [
                self.house_dict[self.G.nodes[node]["household"]]["isolated_time"] - self.G.nodes[node]["symptom_onset"] 
                for node in self.G.nodes() 
                if self.G.nodes[node]["isolated"] == True
            ]
        else:
            return [
                self.house_dict[self.G.nodes[node]["household"]]["isolated_time"] - self.G.nodes[node]["symptom_onset"] 
                for node in self.G.nodes() 
                if self.G.nodes[node]["isolated"] == True
                and self.house_dict[self.G.nodes[node]["household"]]["being_contact_traced_from"] != None
            ]


    def infected_to_isolation_times(self, include_self_reports = True):
        if include_self_reports:
            return [
                self.house_dict[self.G.nodes[node]["household"]]["isolated_time"] - self.G.nodes[node]["time_infected"] 
                for node in self.G.nodes() 
                if self.G.nodes[node]["isolated"] == True
            ]
        else:
            return [
                self.house_dict[self.G.nodes[node]["household"]]["isolated_time"] - self.G.nodes[node]["time_infected"] 
                for node in self.G.nodes() 
                if self.G.nodes[node]["isolated"] == True
                and self.house_dict[self.G.nodes[node]["household"]]["being_contact_traced_from"] != None
            ]

    def get_cmap(self, n, name='hsv'):
        '''
        Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.
        '''
        return plt.cm.get_cmap(name, n)

    def make_proxy(self, clr, **kwargs):
        """[summary]

        Arguments:
            clr {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        return Line2D([0, 1], [0, 1], color=clr, **kwargs)

    def node_colour(self, node):
        isolation_status = self.G.nodes[node]["isolated"]
        contact_traced = self.G.nodes[node]["contact_traced"]
        if isolation_status is True:
            return "yellow"
        elif contact_traced is True:
            return "orange"
        else:
            return "white"

    def draw_network(self):

        node_colour_map = [self.node_colour(node) for node in self.G.nodes()]

        # The following chunk of code draws the pretty branching processes
        edge_colour_map = [self.G.edges[edge]["colour"] for edge in self.G.edges()]

        # Legend for explaining edge colouring
        proxies = [self.make_proxy(clr, lw=1) for clr in (self.default_edge_colour,
                                                    self.contact_traced_edge_colour_within_house,
                                                    self.contact_traced_edge_between_house,
                                                    self.app_traced_edge,
                                                    self.failed_contact_tracing)]
        labels = ("Transmission, yet to be traced",
                "Within household contact tracing",
                "Between household contact tracing",
                "App traced edge",
                "Failed contact trace")

        node_households = {}
        for node in self.G.nodes():
            house = self.G.nodes[node]["household"]
            node_households.update({node: house})

        pos = graphviz_layout(self.G, prog='twopi')
        plt.figure(figsize=(8, 8))

        nx.draw(
            self.G, pos, node_size=150, alpha=0.9, node_color=node_colour_map, edge_color=edge_colour_map,
            labels=node_households
        )
        plt.axis('equal')
        plt.title("Household Branching Process with Contact Tracing")
        plt.legend(proxies, labels)


class model_calibration(household_sim_contact_tracing):

    def gen_mu_local_house(self, house_size):
        """
        Generates an observation of the number of members of each generation in a local (household) epidemic.

        The definition is specific, see the paper by Pellis et al.

        Brief description:
        1) Pretend every node is infected
        2) Draw edges from nodes to other nodes in household if they make an infective contact
        3) The generation of a node is defined as the shotest path length from node 0
        4) Get the vector V where v_i is the number of members of generation i
        """

        # Set up the graph
        G = nx.DiGraph()
        G.add_nodes_from(range(house_size))

        # If the house size is 1 there is no local epidemic
        if house_size == 1:
            return [1]

        # Loop over every node in the household
        for node in G.nodes():

            # Other nodes in house
            other_nodes = [member for member in range(house_size) if member != node]

            # Get the infectious period:
            effective_infectious_period = 21

            for day in range(1, effective_infectious_period+1):

                # How many infectious contact does the node make
                prob = current_prob_infection(day, self.haz_rate_scale)
                contacts = self.contacts_made_today(house_size)
                contacts_within_house = npr.binomial(contacts, self.proportion_of_within_house_contacts[house_size-1])
                contacts_within_house = min(house_size-1, contacts_within_house)
                infectious_contacts = npr.binomial(contacts_within_house, prob)

                # Add edges to the graph based on who was contacted with an edge
                infected_nodes = npr.choice(other_nodes, infectious_contacts, replace=False)

                for infected in infected_nodes:
                    G.add_edge(node, infected)

        # Compute the generation of each node
        generations = []
        for node in G.nodes:
            if nx.has_path(G, 0, node) is True:
                generations.append(nx.shortest_path_length(G, 0, node))

        # Work of the size of each generation
        mu_local = []
        for gen in range(house_size):
            mu_local.append(sum([int(generations[i] == gen) for i in range(len(generations))]))

        return mu_local

    def estimate_mu_local(self):
        """
        Computes the expected size of each generation for a within household epidemic by simulation
        """

        repeats = 1000
        mu_local = np.array([0.]*6)

        # Loop over the possible household sizes
        for house_size in range(1, 7):
            mu_local_house = np.array([0]*6)

            # Generate some observations of the size of each generation and keep adding to an empty array
            for _ in range(repeats):
                sample = self.gen_mu_local_house(house_size)
                sample = np.array(sample + [0]*(6 - len(sample)))
                mu_local_house += sample

            # Get the mean
            mu_local_house = mu_local_house/repeats

            # Normalize by the size-biased distribution (prob of household size h * house size h) and the normalized to unit probability
            update = mu_local_house*self.size_biased_distribution[house_size - 1]
            mu_local += update

        return mu_local

    def estimate_mu_global(self):
        "Performs a Monte-Carlo simulation of the number of global infections for a given house and generation"
        repeats = 1000

        total_infections = 0
        for _ in range(repeats):

            # Need to use size-biased distribution here maybe?
            house_size = self.size_of_household()

            effective_infectious_period = 21

            for day in range(0, effective_infectious_period + 1):

                # How many infectious contact does the node make
                prob = current_prob_infection(day, self.haz_rate_scale)
                contacts = self.contacts_made_today(house_size)
                contacts_within_house = npr.binomial(contacts, self.proportion_of_within_house_contacts[house_size-1])
                contacts_within_house = min(house_size-1, contacts_within_house)
                contacts_outside_house = contacts - contacts_within_house
                infectious_contacts = npr.binomial(contacts_outside_house, prob)
                total_infections += infectious_contacts

        return total_infections/repeats

    def calculate_R0(self):
        """
        The following function calculates R_0 for a given (alpha, p_inf) pair, using the method described in the paper by Pellis et. al.
        """

        mu_global = self.estimate_mu_global()

        mu_local = self.estimate_mu_local()

        g = lambda x: 1-sum([mu_global*mu_local[i]/(x**(i+1)) for i in range(6)])
        output = s.optimize.root_scalar(g, x0=1, x1=4)
        return output.root

    def estimate_secondary_attack_rate(self):
        """Simulates a household epidemic, with a single starting case. Outside household infections are performed but will not propagate.
        """

        # Reset the simulation to it's initial state
        self.reset_simulation()

        # Initial households are allowed to run the household epidemics
        starting_households = list(range(1, self.starting_infections))

        while len(self.active_infections) is not 0:

            # Increment the infection process
            self.increment_infection()

            # recover nodes that need it
            self.perform_recoveries()

            # set any node that was an outside-household infection to the recovered state, so that they are not simulated.
            [
                self.G.nodes[node].update({"recovered": True})
                for node in self.G.nodes()
                if self.G.nodes[node]["household"] not in starting_households
                and self.G.nodes[node]["recovered"] is False
            ]

            self.time += 1

        total_infected = sum([
            len(self.house_dict[house]["nodes"]) - 1
            for house in starting_households
        ])

        total_exposed = sum([
            self.house_dict[house]["size"] - 1
            for house in starting_households
        ])

        return total_infected/total_exposed
