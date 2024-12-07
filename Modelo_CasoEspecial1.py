from pyomo.environ import *
import pandas as pd

# Lectura de datos
clients_df = pd.read_csv('vrp_case_data/case_5_recharge_nodes/Clients.csv')
depots_df = pd.read_csv('vrp_case_data/case_5_recharge_nodes/Depots.csv')
vehicles_df = pd.read_csv('vrp_case_data/case_5_recharge_nodes/Vehicles.csv')
recharge_nodes_df = pd.read_csv('vrp_case_data/case_5_recharge_nodes/RechargeNodes.csv')
depot_capacities_df = pd.read_csv('vrp_case_data/case_5_recharge_nodes/DepotCapacities.csv')

# Crear el conjunto combinado de nodos (clientes + depósitos + nodos de recarga)
clients_df['ClientID'] = clients_df['ClientID'].apply(lambda x: f"NCliente{x}")
depots_df['DepotID'] = depots_df['DepotID'].apply(lambda x: f"NBodega{x}")
recharge_nodes_df['RechargeNodeID'] = recharge_nodes_df['RechargeNodeID'].apply(lambda x: f"NRecarga{x}")

nodes = clients_df['ClientID'].tolist() + depots_df['DepotID'].tolist() + recharge_nodes_df['RechargeNodeID'].tolist()

# Procesar capacidades de depósitos
depot_capacities_df['DepotID'] = depot_capacities_df['DepotID'].apply(lambda x: f"NBodega{x}")
depot_capacities_df['Product'].fillna(0, inplace=True)

# Parámetros asumidos para el tiempo y costo de recarga
vehicles_df['BatteryCapacity'] = vehicles_df['Range'] / 10
recharge_nodes_df['RechargeRate'] = 5
recharge_nodes_df['RechargeCost'] = 500

# Modelo
model = ConcreteModel()

# Conjuntos
model.N = Set(initialize=nodes, doc="Nodos (clientes + depósitos + nodos de recarga)")
model.V = Set(initialize=vehicles_df['VehicleType'].tolist(), doc="Tipos de vehículos")
model.R = Set(initialize=recharge_nodes_df['RechargeNodeID'].tolist(), doc="Nodos de recarga")

# Parámetros
def get_coordinates(df, id_column):
    return df.set_index(id_column)[['Longitude', 'Latitude']].T.to_dict('list')

coordinates = {**get_coordinates(clients_df, 'ClientID'),
               **get_coordinates(depots_df, 'DepotID'),
               **get_coordinates(recharge_nodes_df, 'RechargeNodeID')}
model.coordinates = Param(model.N, initialize=coordinates, doc="Coordenadas de los nodos", within=Any)

# Distancia euclidiana entre nodos
def distance(o, d):
    x1, y1 = coordinates[o]
    x2, y2 = coordinates[d]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

model.distances = Param(model.N, model.N, initialize=lambda model, o, d: distance(o, d) if o != d else 0)

# Costos de recarga y tiempos
recharge_cost = recharge_nodes_df.set_index('RechargeNodeID')['RechargeCost'].to_dict()
recharge_rate = recharge_nodes_df.set_index('RechargeNodeID')['RechargeRate'].to_dict()
model.recharge_cost = Param(model.R, initialize=recharge_cost, doc="Tarifa de recarga en COP/kWh")
model.recharge_rate = Param(model.R, initialize=recharge_rate, doc="Tasa de recarga en kWh/min")

# Variables
model.y = Var(model.N, model.N, model.V, domain=Binary, doc="Flujo de vehículos entre nodos")
model.u = Var(model.N, model.V, domain=NonNegativeReals, doc="Subtour elimination")

# Función objetivo
def objective_rule(model):
    return sum(
        model.distances[o, d] * model.y[o, d, v]
        for o in model.N for d in model.N for v in model.V
    ) + sum(
        model.recharge_cost[r] * sum(model.y[r, d, v] for d in model.N) * vehicles_df.loc[vehicles_df['VehicleType'] == v, 'BatteryCapacity'].values[0]
        for r in model.R for v in model.V
    ) + sum(
        (vehicles_df.loc[vehicles_df['VehicleType'] == v, 'BatteryCapacity'].values[0] / model.recharge_rate[r]) * sum(model.y[r, d, v] for d in model.N)
        for r in model.R for v in model.V
    )

model.objective = Objective(rule=objective_rule, sense=minimize)

# Restricciones
def flow_balance_rule(model, n, v):
    if n in clients_df['ClientID'].tolist():
        return sum(model.y[o, n, v] for o in model.N if o != n) == \
               sum(model.y[n, d, v] for d in model.N if d != n)
    return Constraint.Skip

model.flow_balance = Constraint(model.N, model.V, rule=flow_balance_rule)

def vehicle_entry_rule(model, n):
    if n in clients_df['ClientID'].tolist():
        return sum(model.y[o, n, v] for o in model.N for v in model.V if o != n) == 1
    return Constraint.Skip

model.vehicle_entry = Constraint(model.N, rule=vehicle_entry_rule)

# Subtour elimination (MTZ)
def subtour_elimination_rule(model, i, j, v):
    if i != j and i in nodes and j in nodes:
        return model.u[i, v] - model.u[j, v] + len(nodes) * model.y[i, j, v] <= len(nodes) - 1
    return Constraint.Skip

model.subtour_elimination = Constraint(model.N, model.N, model.V, rule=subtour_elimination_rule)

# Restricción de capacidad de los vehículos
def capacity_rule(model, v):
    return sum(
        clients_df.loc[clients_df['ClientID'] == c, 'Product'].values[0] * model.y[c, d, v]
        for c in clients_df['ClientID'].tolist()
        for d in model.N if c != d
    ) <= vehicles_df.loc[vehicles_df['VehicleType'] == v, 'Capacity'].values[0]

model.capacity_constraint = Constraint(model.V, rule=capacity_rule)

# Resolución
solver = SolverFactory('glpk')
solver.solve(model, tee=True, timelimit=120)

# Generar rutas con etiquetas
routes = []
for v in model.V:
    for o in model.N:
        for d in model.N:
            if model.y[o, d, v].value == 1:
                routes.append([v, o, d])

routes_df = pd.DataFrame(routes, columns=['ID-Vehiculo', 'ID-Origen', 'ID-Destino'])
routes_df.to_csv("./rutas/grupo8-caso-especial-1-ruta.csv", index=False)

print("Archivo generado: rutas/grupo8-caso-especial-1-ruta.csv")
