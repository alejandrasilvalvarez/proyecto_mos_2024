# Caso de Gestión de oferta

from pyomo.environ import *
import pandas as pd

# ------------------
# Lectura de datos
# ------------------
clients_df = pd.read_csv('vrp_case_data/case_4_multi_product/Clients.csv')
depots_df = pd.read_csv('vrp_case_data/case_4_multi_product/Depots.csv')
vehicles_df = pd.read_csv('vrp_case_data/case_4_multi_product/Vehicles.csv')
# Crear el conjunto combinado de nodos (clientes + depósitos)
# Diferenciar los IDs de los nodos al cargar los datos
clients_df['ClientID'] = clients_df['ClientID'].apply(lambda x: f"NCliente{x}")
depots_df['DepotID'] = depots_df['DepotID'].apply(lambda x: f"NBodega{x}")
nodes = clients_df['ClientID'].tolist() + depots_df['DepotID'].tolist()
#capacidades de la bodega
depot_capacities_df = pd.read_csv('vrp_case_data/case_4_multi_product/DepotCapacities.csv')
# Formatear los IDs de los depósitos para que coincidan con el modelo
depot_capacities_df['DepotID'] = depot_capacities_df['DepotID'].apply(lambda x: f"NBodega{x}")
# Conjunto de tipos de productos
product_types = ['Product-Type-A', 'Product-Type-B', 'Product-Type-C']

# ------------------
#  Modelo 
# ------------------
model = ConcreteModel()

# ------------------
# Conjuntos
# ------------------

model.N = Set(initialize=nodes, doc="Nodos (clientes + depósitos)")
model.V = Set(initialize=vehicles_df['VehicleType'].unique().tolist(), doc="Tipos de vehículos")
model.P = Set(initialize=product_types, doc="Tipos de productos")

# ------------------
# Parámetros
# ------------------

def get_coordinates(df, id_column):
    return df.set_index(id_column)[['Longitude', 'Latitude']].T.to_dict('list')

coordinates = {**get_coordinates(clients_df, 'ClientID'), **get_coordinates(depots_df, 'DepotID')}
model.coordinates = Param(model.N, initialize=coordinates, doc="Coordenadas de los nodos", within=Any)

# Distancia euclidiana entre nodos
def distance(o, d):
    x1, y1 = coordinates[o]
    x2, y2 = coordinates[d]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

model.distances = Param(model.N, model.N, initialize=lambda model, o, d: distance(o, d) if o != d else 0)

# Demanda de productos por cliente
def initialize_demand(model, c, p):
    if c in clients_df['ClientID'].tolist():
        return clients_df.loc[clients_df['ClientID'] == c, p].values[0]
    return 0  # Si no hay demanda, asignar 0

model.demand = Param(model.N, model.P, initialize=initialize_demand, doc="Demanda de productos por cliente", within=NonNegativeReals)



# ------------------
# Variables
# ------------------
model.y = Var(model.N, model.N, model.V, domain=Binary, doc="Flujo de vehículos entre nodos")
model.u = Var(model.N, model.V, domain=NonNegativeReals, doc="Subtour elimination")

# ------------------
# Función objetivo
# ------------------
def objective_rule(model):
    return sum(
        model.distances[o, d] * model.y[o, d, v]
        for o in model.N for d in model.N for v in model.V
    )

model.objective = Objective(rule=objective_rule, sense=minimize)

# ------------------
# Restricciones
# ------------------
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
    if i != j and i in clients_df['ClientID'].tolist() and j in clients_df['ClientID'].tolist():
        return model.u[i, v] - model.u[j, v] + len(nodes) * model.y[i, j, v] <= len(nodes) - 1
    return Constraint.Skip

model.subtour_elimination = Constraint(model.N, model.N, model.V, rule=subtour_elimination_rule)

# Restricción de capacidad de los vehículos
def capacity_rule(model, v):
    # Capacidad total utilizada por el vehículo para todos los productos
    return sum(
        model.demand[c, p] * sum(model.y[c, d, v] for d in model.N if c != d)
        for c in clients_df['ClientID'].tolist()
        for p in model.P
    ) <= vehicles_df.loc[vehicles_df['VehicleType'] == v, 'Capacity'].values[0]


model.capacity_constraint = Constraint(model.V, rule=capacity_rule)


# Capacidad de productos por depósito
def initialize_depot_capacity(model, d, p):
    if d in depot_capacities_df['DepotID'].values:
        return depot_capacities_df.loc[depot_capacities_df['DepotID'] == d, p].values[0]
    return 0  # Si no hay capacidad especificada, se asume 0

model.depot_capacity = Param(model.N, model.P, initialize=initialize_depot_capacity, doc="Capacidad de productos por depósito", within=NonNegativeReals)

# Restricción de capacidad en depósitos
def depot_capacity_rule(model, d, p):
    if d in depots_df['DepotID'].tolist():
        return sum(
            model.demand[c, p] * sum(model.y[c, d, v] for v in model.V)
            for c in clients_df['ClientID'].tolist()
        ) <= model.depot_capacity[d, p]
    return Constraint.Skip

model.depot_capacity_constraint = Constraint(model.N, model.P, rule=depot_capacity_rule)

# Restricción de capacidad en vehículos
def vehicle_capacity_rule(model, v):
    return sum(
        model.demand[c, p] * sum(model.y[c, d, v] for d in model.N)
        for c in clients_df['ClientID'].tolist() for p in model.P
    ) <= vehicles_df.loc[vehicles_df['VehicleType'] == v, 'Capacity'].values[0]

model.vehicle_capacity_constraint = Constraint(model.V, rule=vehicle_capacity_rule)



# ------------------
# Resolución
# ------------------
solver = SolverFactory('glpk')
solver.solve(model, tee=True)

# ------------------
# Guardar resultados
# ------------------
# Generar rutas con detalle de productos
routes = []
for v in model.V:
    for o in model.N:
        for d in model.N:
            if model.y[o, d, v].value == 1:
                transported_products = {p: sum(model.demand[o, p] for p in model.P) for p in model.P}
                routes.append([v, o, d, transported_products])

# ------------------
# Guardar en archivo CSV
# ------------------
routes_df = pd.DataFrame(routes, columns=['ID-Vehiculo', 'ID-Origen', 'ID-Destino', 'Productos-Transportados'])
routes_df.to_csv("./rutas/grupo8-caso-escenarioprueba-4-ruta.csv", index=False)

print("Archivo generado: rutas/grupo8-caso-escenarioprueba-4-ruta.csv")