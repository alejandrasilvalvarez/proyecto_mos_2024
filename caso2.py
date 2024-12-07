import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
import requests
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from math import radians, sin, cos, sqrt, atan2

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

# Combinar coordenadas de depósitos y clientes para usar en OSRM
all_coords = list(zip(depots["Longitude"], depots["Latitude"])) + list(zip(clients["Longitude"], clients["Latitude"]))


# Función para obtener las distancias usando OSRM
def osrm_distance(coords):
    """
    Calcula las distancias y duraciones usando el servicio OSRM para vehículos terrestres.
    """
    coords_str = ';'.join([f"{lon},{lat}" for lon, lat in coords])

    url = f"https://router.project-osrm.org/table/v1/driving/{coords_str}"
    params = {
        'sources': ';'.join(map(str, range(len(coords)))),
        'destinations': ';'.join(map(str, range(len(coords)))),
        'annotations': 'distance,duration'
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise RuntimeError(f"OSRM request failed: {response.status_code}, {response.text}")

    data = response.json()
    return np.array(data['distances'])/1000, np.array(data['durations'])/60


# Cargar matrices de distancias y duraciones desde OSRM
osrm_distance_matrix, osrm_duration_matrix = osrm_distance(all_coords)

# Extraer submatrices para depósitos y clientes
num_depots = len(depots)
num_clients = len(clients)

osrm_distances = osrm_distance_matrix[:num_depots, num_depots:]
osrm_durations = osrm_duration_matrix[:num_depots, num_depots:]


# Función Haversine para distancias en línea recta
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en kilómetros
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # Distancia en kilómetros


# Inicializar distancias entre depósitos y clientes
def initialize_distances(model, d, c, v):
    depot_idx = depots[depots["DepotID"] == d].index[0]
    client_idx = clients[clients["ClientID"] == c].index[0]

    if v in ["Gas Car", "Ev"]:
        # Usar OSRM para vehículos terrestres
        return osrm_distances[depot_idx, client_idx]
    elif v == "Drone":
        # Usar Haversine para vehículos aéreos
        depot_coords = (depots.loc[depot_idx, "Latitude"], depots.loc[depot_idx, "Longitude"])
        client_coords = (clients.loc[client_idx, "Latitude"], clients.loc[client_idx, "Longitude"])
        return haversine(depot_coords[0], depot_coords[1], client_coords[0], client_coords[1])


model.distances = Param(model.D, model.C, model.V, initialize=initialize_distances, within=NonNegativeReals)

# Parámetros de capacidad y rango
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
                distance_cost = freight_rate[v] * model.distances[d, c, v] * model.x[d, c, v]
                time_cost = time_rate[v] * (model.distances[d, c, v] / 60) * model.x[d, c, v]
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
    return model.distances[d, c, v] * model.x[d, c, v] <= model.range[v]

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

# Mostrar matriz de costos
cost_matrix = []
for d in model.D:
    for c in model.C:
        for v in model.V:
            distance_cost = freight_rate[v] * model.distances[d, c, v]
            time_cost = time_rate[v] * (model.distances[d, c, v] / 60)
            maintenance_cost = daily_maintenance[v]
            total_cost = distance_cost + time_cost + maintenance_cost
            cost_matrix.append([d, c, v, total_cost])

cost_df = pd.DataFrame(cost_matrix, columns=["DepotID", "ClientID", "VehicleType", "TotalCost"])
print("\nMatriz de Costos:")
print(cost_df)
