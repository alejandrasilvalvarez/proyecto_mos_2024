import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
from math import radians, sin, cos, sqrt, atan2

# Leer los archivos
clients = pd.read_csv("vrp_case_data/case_2_cost/clients.csv")
depots = pd.read_csv("vrp_case_data/case_2_cost/depots.csv")
vehicles = pd.read_csv("vrp_case_data/case_2_cost/vehicles.csv")

# Normalizar los nombres de los vehículos para evitar inconsistencias
vehicles["VehicleType"] = vehicles["VehicleType"].str.strip().str.title()

# Diccionarios de costos ajustados
freight_rate = {"Gas Car": 5000, "Drone": 500, "Ev": 4000}
time_rate = {"Gas Car": 500, "Drone": 500, "Ev": 500}
daily_maintenance = {"Gas Car": 30000, "Drone": 3000, "Ev": 21000}

# Crear el modelo
model = ConcreteModel()

# Conjuntos
model.D = Set(initialize=depots["DepotID"].unique())
model.C = Set(initialize=clients["ClientID"].unique())
model.V = Set(initialize=vehicles["VehicleType"].unique())

# Función para calcular distancias usando la fórmula del Haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en kilómetros
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # Distancia en kilómetros

# Inicializar las distancias entre cada depósito y cliente
def initialize_distances(model, d, c):
    depot = depots.loc[depots["DepotID"] == d]
    client = clients.loc[clients["ClientID"] == c]
    depot_coords = (depot["Latitude"].iloc[0], depot["Longitude"].iloc[0])
    client_coords = (client["Latitude"].iloc[0], client["Longitude"].iloc[0])
    return haversine(depot_coords[0], depot_coords[1], client_coords[0], client_coords[1])

model.distances = Param(model.D, model.C, initialize=initialize_distances, within=NonNegativeReals)

# Parámetros
model.capacity = Param(model.V, initialize={v: vehicles.loc[vehicles["VehicleType"] == v, "Capacity"].iloc[0] for v in model.V})
model.range = Param(model.V, initialize={v: vehicles.loc[vehicles["VehicleType"] == v, "Range"].iloc[0] for v in model.V})

# Variables
model.x = Var(model.D, model.C, model.V, domain=Binary)

# Función de costo
def cost_function(model):
    total_cost = 0
    for d in model.D:
        for c in model.C:
            for v in model.V:
                distance_cost = freight_rate[v] * model.distances[d, c] * model.x[d, c, v]
                time_cost = time_rate[v] * (model.distances[d, c] / 60) * model.x[d, c, v]  # Asumiendo 60 km/h promedio
                maintenance_cost = daily_maintenance[v] * model.x[d, c, v]
                total_cost += distance_cost + time_cost + maintenance_cost
    return total_cost

model.obj = Objective(rule=cost_function, sense=minimize)

# Restricción de capacidad
def capacity_constraint(model, c):
    return sum(model.x[d, c, v] * model.capacity[v] for d in model.D for v in model.V) >= 1

model.capacity_constraint = Constraint(model.C, rule=capacity_constraint)

# Restricción de rango
def range_constraint(model, d, c, v):
    return model.distances[d, c] * model.x[d, c, v] <= model.range[v]

model.range_constraint = Constraint(model.D, model.C, model.V, rule=range_constraint)

# Resolver el modelo
opt = SolverFactory('glpk')
results = opt.solve(model)

# Mostrar resultados
print("Costo total:", model.obj())
for d in model.D:
    for c in model.C:
        for v in model.V:
            if model.x[d, c, v].value > 0.5:
                print(f"Depot {d} entrega al Cliente {c} usando el Vehículo {v}")

# Generar la matriz de costos
cost_matrix = []
for d in model.D:
    for c in model.C:
        for v in model.V:
            distance_cost = freight_rate[v] * model.distances[d, c]
            time_cost = time_rate[v] * (model.distances[d, c] / 60)
            maintenance_cost = daily_maintenance[v]
            total_cost = distance_cost + time_cost + maintenance_cost
            cost_matrix.append([d, c, v, total_cost])

# Convertir la matriz en un DataFrame para mejor visualización
cost_df = pd.DataFrame(cost_matrix, columns=["DepotID", "ClientID", "VehicleType", "TotalCost"])
print("\nMatriz de Costos:")
print(cost_df)
