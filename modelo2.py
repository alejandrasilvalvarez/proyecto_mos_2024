from pyomo.environ import *
import pandas as pd

# lectura datos

clients_df = pd.read_csv('vrp_case_data/case_1_base/clients.csv')
depots_df = pd.read_csv('vrp_case_data/case_1_base/depots.csv')
drone_only_df = pd.read_csv('vrp_case_data/case_1_base/drone_only.csv')
ev_only_df = pd.read_csv('vrp_case_data/case_1_base/ev_only.csv')
gas_car_only_df = pd.read_csv('vrp_case_data/case_1_base/gas_car_only.csv')
vehicles_df = pd.read_csv('vrp_case_data/case_1_base/vehicles.csv')

#  Modelo 
model = ConcreteModel()

#  Conjuntos 
# Conjunto de clientes
model.C = Set(initialize=clients_df['ClientID'].tolist(), doc="Conjunto de clientes")
# Conjunto de depósitos
model.D = Set(initialize=depots_df['DepotID'].tolist(), doc="Conjunto de depósitos")
# Conjunto de tipos de vehículos (drones, EV, Gas Car)
model.V = Set(initialize=vehicles_df['VehicleType'].unique().tolist(), doc="Tipos de vehículos")

#  Parámetros 
# Coordenadas de los clientes
model.client_coordinates = Param(
    model.C,
    initialize=clients_df.set_index('ClientID')[['Longitude', 'Latitude']].T.to_dict('list'),
    doc="Coordenadas de los clientes",
    within=Any,
)

# Coordenadas de los depósitos
model.depot_coordinates = Param(
    model.D,
    initialize=depots_df.set_index('DepotID')[['Longitude', 'Latitude']].T.to_dict('list'),
    doc="Coordenadas de los depósitos",
    within=Any,
)

# Datos de vehículos (capacidad y rango)
model.vehicle_data = Param(
    model.V,
    initialize={
        **drone_only_df.set_index('VehicleType')[['Capacity', 'Range']].T.to_dict('list'),
        **ev_only_df.set_index('VehicleType')[['Capacity', 'Range']].T.to_dict('list'),
        **gas_car_only_df.set_index('VehicleType')[['Capacity', 'Range']].T.to_dict('list'),
    },
    doc="Capacidad y rango de los vehículos",
    within=Any,
)

#  Variables 
# Variable binaria para asignar clientes a depósitos
model.x = Var(model.C, model.D, model.V, domain=Binary, doc="Asignación de cliente a depósito y vehículo")

#  Función Objetivo 
# Minimizar la distancia total recorrida por los vehículos
def objective_rule(model):
    return sum(
        model.x[c, d, v] * (
            (model.client_coordinates[c][0] - model.depot_coordinates[d][0])**2 + 
            (model.client_coordinates[c][1] - model.depot_coordinates[d][1])**2)**0.5
        for c in model.C for d in model.D for v in model.V
    )

model.objective = Objective(rule=objective_rule, sense=minimize)

#  Restricciones 
# Restricción de asignación de vehículos a clientes
def assignment_rule(model, c):
    return sum(model.x[c, d, v] for d in model.D for v in model.V) == 1  # Cada cliente debe ser asignado a un vehículo

model.assignment_constraint = Constraint(model.C, rule=assignment_rule)

# Restricción de capacidad de los vehículos
def capacity_rule(model, d, v):
    return sum(model.x[c, d, v] * clients_df.loc[clients_df['ClientID'] == c, 'Product'].values[0] 
               for c in model.C) <= model.vehicle_data[v][0]  # Capacidad del vehículo no excedida

model.capacity_constraint = Constraint(model.D, model.V, rule=capacity_rule)

# solucion
solver = SolverFactory('glpk') 
solver.solve(model, tee=True)


for c in model.C:
    for d in model.D:
        for v in model.V:
            if model.x[c, d, v].value == 1:
                print(f"Cliente {c} asignado al depósito {d} con vehículo {v}")
