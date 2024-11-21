from pyomo.environ import *

# Crear el modelo
model = ConcreteModel()

# === 1. Definir conjuntos ===
# Tipos de vehículos
model.VT = Set(initialize=['VT1', 'VT2', 'VT3'])  # Vehículos tradicionales              # Drones de alto alcance
model.V = Set(initialize=model.VT)  # Todos los vehículos

# Clientes, entregas y depósitos
model.C = Set(initialize=['C1', 'C2', 'C3'])      # Clientes
model.E = Set(initialize=['E1', 'E2', 'E3'])      # Entregas
model.CD = Set(initialize=['Depot'])              # Centros de distribución
model.EC = Set(initialize=['EC1', 'EC2'])         # Estaciones de carga

# Días
model.D = Set(initialize=[1, 2, 3])  # 3 días de operación como ejemplo

# === 2. Definir parámetros ===
# Parámetros básicos
model.distanciaMax = Param(initialize=150)  # Distancia máxima por viaje (km)
model.distanciaProm = Param(initialize=70)  # Distancia promedio diaria (km)
model.demandaDiaria = Param(initialize=20000)  # Total de entregas diarias
model.uDeposito = Param(initialize='Bogotá')  # Depósito principal

# Pesos asociados a cada entrega
peso_carga_data = {'E1': 10, 'E2': 15, 'E3': 20}
model.pesoCarga = Param(model.E, initialize=peso_carga_data)

# Tiempo de viaje por día para cada vehículo
tiempo_diario_data = {('VT1', 1): 6, ('VT1', 2): 5, ('VT1', 3): 7,
                      ('VT2', 1): 8, ('VT2', 2): 6, ('VT2', 3): 7,
                      ('VT3', 1): 7, ('VT3', 2): 5, ('VT3', 3): 6}
model.tiempoDiario = Param(model.V, model.D, initialize=tiempo_diario_data)

# Tiempo de recarga para cada vehículo por día (en horas)
tiempo_recarga_data = {('VT1', 1): 1, ('VT2', 1): 1.5, ('VT3', 1): 2,
                       ('VT1', 2): 1, ('VT2', 2): 1.5, ('VT3', 2): 2,
                       ('VT1', 3): 1, ('VT2', 3): 1.5, ('VT3', 3): 2}
model.T = Param(model.V, model.D, initialize=tiempo_recarga_data)

# Emisiones de CO₂ por kilómetro para cada vehículo (en kg CO2/km)
emisiones_data = {'VT1': 0.2, 'VT2': 0.15, 'VT3': 0.1}
model.emisiones = Param(model.V, initialize=emisiones_data)

# Capacidades y demandas
capacity = {'VT1': 50, 'VT2': 70, 'VT3': 100}
model.capacity = Param(model.V, initialize=capacity)
demand = {'C1': 30, 'C2': 40, 'C3': 50}
model.demand = Param(model.C, initialize=demand)

# Costos
model.costoCarga = Param(model.V, initialize={'VT1': 10, 'VT2': 8, 'VT3': 6})
model.costoDistancia = Param(model.V, initialize={'VT1': 5, 'VT2': 4, 'VT3': 3})
model.costoTiempo = Param(model.V, initialize={'VT1': 2, 'VT2': 1.5, 'VT3': 1})
model.costoRecarga = Param(model.V, initialize={'VT1': 15, 'VT2': 12, 'VT3': 10})

# Costos de mantenimiento por vehículo y día
costo_mantenimiento_data = {(1, 'VT1'): 50, (1, 'VT2'): 40, (1, 'VT3'): 30,
                            (2, 'VT1'): 50, (2, 'VT2'): 40, (2, 'VT3'): 30,
                            (3, 'VT1'): 50, (3, 'VT2'): 40, (3, 'VT3'): 30}
model.costoMantenimiento = Param(model.D, model.V, initialize=costo_mantenimiento_data)

# Distancias entre nodos
distances = {('Depot', 'C1'): 10, ('Depot', 'C2'): 15, ('Depot', 'C3'): 20,
             ('C1', 'C2'): 5, ('C1', 'C3'): 10, ('C2', 'C3'): 8}
model.distances = Param(model.CD | model.C, model.C, initialize=distances, default=0)


# === 3. Definir variables de decisión ===
# x[e,d,v]: 1 si el vehículo v carga la entrega e en el día d
model.x = Var(model.E, model.D, model.V, domain=Binary)

model.u = Var(model.C, model.V, domain=NonNegativeReals)

# z[i,c,v,d]: 1 si el vehículo v transporta del punto i al cliente c en el día d

# Crear el conjunto de combinaciones válidas para z[i, c, v, d]
valid_z_indices = [
    (i, c, v, d)
    for i in model.CD | model.C
    for c in model.C
    for v in model.V
    for d in model.D
    if i != c
]
model.Z_INDEX = Set(dimen=4, initialize=valid_z_indices)

model.z = Var(model.Z_INDEX, domain=Binary)

# w[v,d]: 1 si el vehículo v necesita hacer mantenimiento el día d
model.w = Var(model.V, model.D, domain=Binary)

# === 4. Definir la función objetivo ===
def objective_rule(model):
    return (
        # Costo de carga de artículos
        sum(model.x[e, d, v] * model.costoCarga[v] for e in model.E for d in model.D for v in model.V)
        +
        # Costo diario por distancia viajada
        sum(model.z[i, j, v, d] * model.distances[i, j] * model.costoDistancia[v]
            for i, j, v, d in model.Z_INDEX)
        +
        # Costo diario del tiempo viajado
        sum(model.z[i, j, v, d] * model.costoTiempo[v] for i, j, v, d in model.Z_INDEX)
        +
        # Costo diario de mantenimiento
        sum(model.w[v, d] * model.costoMantenimiento[d, v] for v in model.V for d in model.D)
    )
model.objective = Objective(rule=objective_rule, sense=minimize)

# === 5. Restricciones ===
# Restricción 1: Capacidad por vehículo
def capacity_constraint(model, v, d):
    return sum(model.x[e, d, v] * model.pesoCarga[e] for e in model.E) <= model.capacity[v]
model.capacity_constraint = Constraint(model.V, model.D, rule=capacity_constraint)

# Restricción 2: Tiempo máximo promedio (48 horas)
def time_constraint(model, v, d):
    return sum(
        model.z[i, j, v, d] * model.tiempoDiario[v, d]
        for i, j, v2, d2 in model.Z_INDEX
        if v2 == v and d2 == d
    ) <= 48
model.time_constraint = Constraint(model.V, model.D, rule=time_constraint)

# Restricción 3: Distancia máxima por entrega
def max_distance_constraint(model, v, d):
    return sum(
        model.z[i, c, v2, d2] * model.distances[i, c]
        for i, c, v2, d2 in model.Z_INDEX
        if v2 == v and d2 == d
    ) <= 150
model.max_distance_constraint = Constraint(model.V, model.D, rule=max_distance_constraint)

# Restricción 4: Capacidad del centro de distribución
def distribution_capacity_constraint(model, i, d):
    return sum(model.z[i, j, v, d] for j in model.C for v in model.V) <= 20000
model.distribution_capacity_constraint = Constraint(model.CD, model.D, rule=distribution_capacity_constraint)

# Restricción 5: Tiempo de carga y descarga
def loading_time_constraint(model, v, d):
    return (
        sum(model.x[e, d, v] * model.T[v, d] for e in model.E)
    ) <= 2
model.loading_time_constraint = Constraint(model.V, model.D, rule=loading_time_constraint)

# Restricción 6: Emisiones de CO2
def co2_emissions_constraint(model, v):
    return sum(model.z[i, c, v, d] * model.distances[i, c] * model.emisiones[v]
               for i, c, v2, d in model.Z_INDEX if v2 == v) <= 50
model.co2_emissions_constraint = Constraint(model.V, rule=co2_emissions_constraint)

# Restricción 7: Demanda diaria
def daily_demand_constraint(model, d):
    return sum(model.z[i, j, v, d] for i, j, v, d2 in model.Z_INDEX if d2 == d) >= 20000
model.daily_demand_constraint = Constraint(model.D, rule=daily_demand_constraint)

# Restricción 8: Mantenimiento (cada 10,000 km)
# Variable para kilómetros acumulados por vehículo y día
model.kilometraje_acumulado = Var(model.V, model.D, domain=NonNegativeReals)

def kilometraje_acumulado_rule(model, v, d):
    if d == 1:
        return model.kilometraje_acumulado[v, d] == sum(
            model.z[i, j, v2, d2] * model.distances[i, j]
            for i, j, v2, d2 in model.Z_INDEX
            if v2 == v and d2 == d
        )
    else:
        return model.kilometraje_acumulado[v, d] == model.kilometraje_acumulado[v, d - 1] + sum(
            model.z[i, j, v2, d2] * model.distances[i, j]
            for i, j, v2, d2 in model.Z_INDEX
            if v2 == v and d2 == d
        )
model.kilometraje_acumulado_constraint = Constraint(model.V, model.D, rule=kilometraje_acumulado_rule)

def mantenimiento_rule(model, v, d):
    return model.kilometraje_acumulado[v, d] <= 10000 * model.w[v, d] + 9999 * (1 - model.w[v, d])
model.maintenance_constraint = Constraint(model.V, model.D, rule=mantenimiento_rule)

# Restricción 9: Continuidad
def continuity_constraint(model, c, v, d):
    entradas = sum(model.z[i, c, v, d] for i, c2, v2, d2 in model.Z_INDEX if c2 == c and v2 == v and d2 == d)
    salidas = sum(model.z[c, k, v, d] for c2, k, v2, d2 in model.Z_INDEX if c2 == c and v2 == v and d2 == d)
    return entradas == salidas
model.continuity_constraint = Constraint(model.C, model.V, model.D, rule=continuity_constraint)

# Restricción 10: Eliminación de subrutas
def subtour_elimination_constraint(model, i, j, v, d):
    if i != j:
        return model.u[i, v] - model.u[j, v] + len(model.C) * model.z[i, j, v, d] <= len(model.C) - 1
    else:
        return Constraint.Skip
model.subtour_elimination_constraint = Constraint(model.C, model.C, model.V, model.D, rule=subtour_elimination_constraint)

# === 6. Resolver el modelo ===
solver = SolverFactory('glpk')
results = solver.solve(model, tee=True)

# Mostrar resultados
if results.solver.termination_condition == TerminationCondition.infeasible:
    print("El modelo no tiene solución factible.")
elif results.solver.termination_condition == TerminationCondition.optimal:
    print("\n==== Resultados ====")
    for v in model.V:
        for d in model.D:
            for e in model.E:
                if model.x[e, d, v].value and model.x[e, d, v].value > 0.5:
                    print(f"Vehículo {v} carga entrega {e} el día {d}")
            for i, j, v2, d2 in model.Z_INDEX:
                if model.z[i, j, v2, d2].value and model.z[i, j, v2, d2].value > 0.5 and v2 == v and d2 == d:
                    print(f"Vehículo {v} transporta de {i} a {j} el día {d}")
            if model.w[v, d].value and model.w[v, d].value > 0.5:
                print(f"Vehículo {v} realiza mantenimiento el día {d}")
    print("Costo total:", model.objective())
else:
    print("El solver no encontró una solución óptima.")

