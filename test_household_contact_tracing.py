import household_contact_tracing as hct    # The code to test
import numpy as np

# generate coverage report using:
# pytest --cov=. --cov-report xml:cov.xml
# in the terminal

test_model = hct.household_sim_contact_tracing(
    haz_rate_scale=0.805,
    contact_tracing_success_prob=0.66,
    contact_trace_delay_par=2,
    overdispersion=0.36,
    infection_reporting_prob=0.8,
    contact_trace=True,
    do_2_step=True,
    test_before_propagate_tracing=True
)


def test_two_step():
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        do_2_step=True
    )

    assert model.max_tracing_index == 2

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
    )

    assert model.max_tracing_index == 1


def test_incubation_period():

    incubation_periods = np.array([
        test_model.incubation_period()
        for i in range(10000)])

    # True mean = 4.83
    assert incubation_periods.mean() < 5
    assert incubation_periods.mean() > 4.5
    # True var = 7.7
    assert incubation_periods.var() > 7.5
    assert incubation_periods.var() < 8.5


def test_testing_delays():

    test_delays = np.array([
        test_model.testing_delay()
        for i in range(10000)])

    # True mean = 1.52 (default)
    assert test_delays.mean() < 1.75
    assert test_delays.mean() > 1.25
    # True var = 1.11
    assert test_delays.var() > 0.8
    assert test_delays.var() < 1.5


def test_reporting_delay():

    reporting_delays = np.array([
        test_model.reporting_delay()
        for i in range(10000)])

    # True mean = 2.68
    assert reporting_delays.mean() < 3
    assert reporting_delays.mean() > 2
    # True var = 2.38^2 = 5.66 ish
    assert reporting_delays.var() > 4
    assert reporting_delays.var() < 7


def test_contacts_made_today():

    house_size = 1
    contacts = np.array(
        [
            test_model.contacts_made_today(house_size)
            for i in range(10000)
        ]
    )

    # true value = 8.87
    assert contacts.mean() < 10.5
    assert contacts.mean() > 8

    house_size = 6
    contacts = np.array(
        [
            test_model.contacts_made_today(house_size)
            for i in range(1000)
        ]
    )

    # true value = 17.69
    assert contacts.mean() < 20
    assert contacts.mean() > 17


def test_contact_trace_delay():

    assert test_model.contact_trace_delay(True) == 0

    trace_delays = np.array(
        [
            test_model.contact_trace_delay(False)
            for i in range(1000)
        ]
    )

    assert trace_delays.mean() < 2.5
    assert trace_delays.mean() > 1.5


def test_new_household():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
    )

    model.time = 100

    model.new_household(new_household_number=10,
                        generation=5,
                        infected_by=6,
                        infected_by_node=3)

    house = model.house_dict[10]

    assert house["size"] in [1, 2, 3, 4, 5, 6]
    assert house["time"] == 100
    assert house["size"] - 1 == house["susceptibles"]
    assert house["generation"] == 5
    assert house["infected_by"] == 6
    assert house["infected_by_node"] == 3


def test_get_edge_between_household():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True)

    # household 1
    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    # infection 1
    model.new_infection(
        node_count=1,
        generation=1,
        household=1)

    # household 2
    model.new_household(
        new_household_number=2,
        generation=2,
        infected_by=1,
        infected_by_node=1)

    # infection 2
    model.new_infection(
        node_count=2,
        generation=2,
        household=2)

    # add an edge between the infections
    model.G.add_edge(1, 2)

    assert model.get_edge_between_household(1, 2) == (1, 2)


def test_is_app_traced():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        prob_has_trace_app=1)

    # household 1
    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    # infection 1
    model.new_infection(
        node_count=1,
        generation=1,
        household=1)

    # household 2
    model.new_household(
        new_household_number=2,
        generation=2,
        infected_by=1,
        infected_by_node=1)

    # infection 2
    model.new_infection(
        node_count=2,
        generation=2,
        household=2)

    # add an edge between the infections
    model.G.add_edge(1, 2)
    assert model.is_edge_app_traced((1, 2))


def test_new_outside_household_infection():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        prob_has_trace_app=1)

    # household 1
    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    # infection 1
    model.new_infection(
        node_count=1,
        generation=1,
        household=1)

    model.new_outside_household_infection(
        infecting_node=1,
        serial_interval=1
    )

    assert model.house_count == 2
    assert model.G.nodes[1]["spread_to"] == [2]
    assert model.G.has_edge(1, 2)


def test_within_household_infection():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        prob_has_trace_app=1)

    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    model.new_infection(
        node_count=1,
        generation=1,
        household=1)

    model.house_dict[1]["house_size"] = 2
    model.house_dict[1]["susceptibles"] = 1

    model.new_within_household_infection(
        infecting_node=1,
        serial_interval=10)

    assert model.house_dict[1]["susceptibles"] == 0
    assert model.G.nodes[1]["spread_to"] == [2]
    assert model.G.nodes[2]["household"] == 1
    assert model.G.nodes[2]["serial_interval"] == 10
    assert model.G.nodes[2]["generation"] == 2
    assert model.G.edges[1, 2]["colour"] == "black"
    assert model.house_dict[1]["within_house_edges"] == [(1, 2)]


def test_perform_recoveries():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        prob_has_trace_app=1)

    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    model.new_infection(
        node_count=1,
        generation=1,
        household=1)

    model.G.nodes[1]["recovery_time"] = 0
    model.perform_recoveries()
    assert model.G.nodes[1]["recovered"] is True


def test_colour_edges_between_houses():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        prob_has_trace_app=1)

    model.new_household(
        new_household_number=1,
        generation=1,
        infected_by=None,
        infected_by_node=None)

    model.new_infection(
        node_count=1,
        generation=1,
        household=1)

    model.node_count = 2

    model.new_outside_household_infection(
        infecting_node=1,
        serial_interval=10
    )

    model.colour_node_edges_between_houses(1, 2, "yellow")
    assert model.G.edges[1, 2]["colour"] == "yellow"


def test_overide_testing_delay():

    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        test_before_propagate_tracing=False)

    assert model.testing_delay() == 0

def test_hh_prob_leave_iso_default():
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        test_before_propagate_tracing=False)
    assert model.hh_propensity_to_leave_isolation() == 0

def test_hh_prob_leave_iso():
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        test_before_propagate_tracing=False,
        hh_prob_propensity_to_leave_isolation=1)
    assert model.hh_propensity_to_leave_isolation() == 1

def test_hh_has_propensity_attr():
    model = hct.household_sim_contact_tracing(
        haz_rate_scale=0.805,
        contact_tracing_success_prob=0.66,
        contact_trace_delay_par=2,
        overdispersion=0.36,
        infection_reporting_prob=0.8,
        contact_trace=True,
        test_before_propagate_tracing=False,
        hh_prob_propensity_to_leave_isolation=0.5)

    assert model.house_dict[1]["propensity_to_leave_isolation"] in (True, False)
