import pyomo.environ as pyomo


class Instance():
    def __init__(self, num_instances, iteration_time, num_stages):
        self.num_instances = num_instances
        self.iteration_time = iteration_time
        self.num_stages = num_stages

num_instances_set = [Instance(4, 1.5, 1), Instance(2, 2.4, 2), Instance(1, 5.2, 3)]
global_num_microbatch = 16


model = pyomo.ConcreteModel()
model.I = pyomo.Set(initialize=list(range(len(num_instances_set))))

T = {
    i: instance.iteration_time
    for i, instance in enumerate(num_instances_set)
}

x = {
    i: instance.num_instances for i, instance in enumerate(num_instances_set)
}
s = {
    i: instance.num_stages
    for i, instance in enumerate(num_instances_set)
}

# Define the Pyomo variable
# nb: number fo microbatches per PipelineSpec
# nb is the variable that need to be solved
model.nb = pyomo.Var(model.I, within=pyomo.PositiveIntegers)

# Objective function
# def objective(model):
#     avg_bT = sum(T[i] / s[i] * model.nb[i] for i in model.I) / len(model.I)
#     return sum((T[i] / s[i] * model.nb[i] - avg_bT) ** 2 for i in model.I)

# model.obj = pyomo.Objective(rule=objective, sense=pyomo.minimize)

# min max objective
model.max_val = pyomo.Var(within=pyomo.PositiveReals)
def max_constraint_rule(model, i):
    return model.max_val >= T[i] / s[i] * model.nb[i]
model.max_constraint = pyomo.Constraint(model.I, rule=max_constraint_rule)
def objective(model):
    return model.max_val
model.obj = pyomo.Objective(rule=objective, sense=pyomo.minimize)


# Define constraints
def c1(model):
    return sum(model.nb[i] * x[i] for i in model.I) == global_num_microbatch


model.constraint1 = pyomo.Constraint(rule=c1)


pyomo.SolverFactory("mindtpy").solve(
    model, mip_solver="glpk", nlp_solver="ipopt", tee=False
)

# check for all i model.nb[i].value is integer, otherwise return None
# 这里返回了None
if not all(model.nb[i].value for i in model.I):
    print("Batch distribution find no results. return None")
else:

    nb_optimal = {
        spec: int(model.nb[i].value)
        for i, spec in zip(model.I, num_instances_set)
    }
    print(f"nb_optimal: {nb_optimal}")
