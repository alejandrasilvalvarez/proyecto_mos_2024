from pyomo.environ import *
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# lectura datos

clients_df = pd.read_csv('vrp_case_data/case_1_base/clients.csv')
depots_df = pd.read_csv('vrp_case_data/case_1_base/depots.csv')
drone_only_df = pd.read_csv('vrp_case_data/case_1_base/drone_only.csv')
ev_only_df = pd.read_csv('vrp_case_data/case_1_base/ev_only.csv')
gas_car_only_df = pd.read_csv('vrp_case_data/case_1_base/gas_car_only.csv')
vehicles_df = pd.read_csv('vrp_case_data/case_1_base/vehicles.csv')

# Función para calcular la distancia Haversine
def haversine(lon1, lat1, lon2, lat2):
    # Convertir grados a radianes
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Fórmula de Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    # Radio de la Tierra en km
    R = 6371.0
    return R * c

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

# Restricción de distancia máxima
def distancia_maxima_rule(model, c, d, v):
    # Coordenadas del cliente
    client_lon, client_lat = model.client_coordinates[c]
    # Coordenadas de la bodega
    depot_lon, depot_lat = model.depot_coordinates[d]
    # Calcular la distancia Haversine
    distancia = haversine(client_lon, client_lat, depot_lon, depot_lat)
    # Restricción: La distancia no debe exceder 150 km
    return model.x[c, d, v] * distancia <= 150

model.distancia_maxima = Constraint(model.C, model.D, model.V, rule=distancia_maxima_rule)

# Capacidad del centro de distribución
def capacidad_deposito_rule(model, d):
    return sum(model.x[c, d, v] for c in model.C for v in model.V) <= 20000
model.capacidad_deposito = Constraint(model.D, rule=capacidad_deposito_rule)

# Solución
solver = SolverFactory('glpk') 
solver.solve(model, tee=True)

# Lista para almacenar las rutas
routes = []

# Iterar sobre las variables y extraer las asignaciones
for c in model.C:
    for d in model.D:
        for v in model.V:
            if model.x[c, d, v].value == 1:  # Si el valor de x es 1, el cliente está asignado a ese depósito y vehículo
                print(f"Cliente {c} asignado al depósito {d} con vehículo {v}")
                routes.append([v, d, c])  # Guardar en la lista: vehículo, depósito, cliente

# Crear un DataFrame con las rutas
routes_df = pd.DataFrame(routes, columns=['ID-Vehiculo', 'ID-Depot', 'ID-Cliente'])

# Guardar en un archivo CSV
output_file = "./rutas/grupo8-caso-escenarioprueba-1-ruta.csv"
routes_df.to_csv(output_file, index=False)

print(f"Archivo generado: {output_file}")
