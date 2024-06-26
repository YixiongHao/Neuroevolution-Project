[NEAT]
fitness_criterion     = mean
fitness_threshold     = 4
pop_size              = 150
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = relu
activation_mutate_rate  = 0.0
activation_options      = relu

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 0.3
bias_max_value          = 5
bias_min_value          = -5
bias_mutate_power       = 0.06
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.0

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.0
conn_delete_prob        = 0.0

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.00

feed_forward            = True
initial_connection      = full_nodirect

# node add/remove rates
node_add_prob           = 0.0
node_delete_prob        = 0.0

# network parameters
num_hidden              = 2
num_inputs              = 2
num_outputs             = 1

# node response options
response_init_mean      = 1
response_init_stdev     = 0
response_max_value      = 1
response_min_value      = 1
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 0.3
weight_max_value        = 5
weight_min_value        = -5
weight_mutate_power     = 0.06
weight_mutate_rate      = 0.7
weight_replace_rate     = 0.0

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2