from pyomo.environ import *

# Crear el modelo
model = ConcreteModel()

# === 1. Definir conjuntos ===
model.V = Set(initialize=['V1', 'V2', 'V3'])  # Vehículos
model.C = Set(initialize=['C1', 'C2', 'C3'])  # Clientes
model.D = Set(initialize=['Depot'])  # Centros de distribución

# === 2. Definir parámetros ===
# Costos por kilómetro
cost_per_km = {'V1': 5, 'V2': 4, 'V3': 3}
model.cost_km = Param(model.V, initialize=cost_per_km)

# Distancias entre nodos
distances = {('Depot', 'C1'): 10, ('Depot', 'C2'): 15, ('Depot', 'C3'): 20,
             ('C1', 'C2'): 5, ('C1', 'C3'): 10, ('C2', 'C3'): 8}
model.distances = Param(model.D | model.C, model.C, initialize=distances, default=0)

# Capacidad máxima de vehículos
capacity = {'V1': 50, 'V2': 70, 'V3': 100}
model.capacity = Param(model.V, initialize=capacity)

# Demanda de clientes
demand = {'C1': 30, 'C2': 40, 'C3': 50}
model.demand = Param(model.C, initialize=demand)

# Tiempos de viaje (horas)
avg_speed = {'V1': 60, 'V2': 50, 'V3': 40}  # Velocidad promedio en km/h
model.time_travel = Param(model.V, initialize=avg_speed, mutable=True)

# Emisiones de CO2 por km (toneladas)
co2_emissions = {'V1': 0.05, 'V2': 0.03, 'V3': 0.02}
model.co2 = Param(model.V, initialize=co2_emissions)

# Límite máximo de emisiones de CO2 por mes
max_emissions = 50
model.max_emissions = Param(initialize=max_emissions)

# === 3. Definir variables de decisión ===
# x[i, j, v] = 1 si el vehículo v va de i a j
model.x = Var(model.D | model.C, model.C, model.V, domain=Binary)

# === 4. Definir la función objetivo ===
def objective_rule(model):
    return sum(model.x[i, j, v] * model.distances[i, j] * model.cost_km[v]
               for i in model.D | model.C for j in model.C for v in model.V)
model.objective = Objective(rule=objective_rule, sense=minimize)

# === 5. Definir restricciones ===

# 5.1 Restricción de capacidad
def capacity_rule(model, v):
    return sum(model.x[i, j, v] * model.demand[j] for i in model.D | model.C for j in model.C) <= model.capacity[v]
model.capacity_constraint = Constraint(model.V, rule=capacity_rule)

# 5.2 Restricción de demanda
def demand_rule(model, j):
    return sum(model.x[i, j, v] for i in model.D | model.C for v in model.V) == 1
model.demand_constraint = Constraint(model.C, rule=demand_rule)

# 5.3 Restricción de tiempo (48 horas máximo)
def time_rule(model, v):
    return sum(model.x[i, j, v] * model.distances[i, j] / model.time_travel[v] for i in model.D | model.C for j in model.C) <= 48
model.time_constraint = Constraint(model.V, rule=time_rule)

# 5.4 Restricción de emisiones de CO2
def co2_rule(model):
    return sum(model.x[i, j, v] * model.distances[i, j] * model.co2[v]
               for i in model.D | model.C for j in model.C for v in model.V) <= model.max_emissions
model.co2_constraint = Constraint(rule=co2_rule)

# 5.5 Restricción de continuidad
def continuity_rule(model, j, v):
    return sum(model.x[i, j, v] for i in model.D | model.C) == sum(model.x[j, k, v] for k in model.C)
model.continuity_constraint = Constraint(model.C, model.V, rule=continuity_rule)

# === 6. Resolver el modelo ===
solver = SolverFactory('glpk')
results = solver.solve(model, tee=True)

# Mostrar resultados
print("\n==== Resultados ====")
for v in model.V:
    for i in model.D | model.C:
        for j in model.C:
            if model.x[i, j, v]() > 0:
                print(f"Vehículo {v} va de {i} a {j}")
print("Costo total:", model.objective())
