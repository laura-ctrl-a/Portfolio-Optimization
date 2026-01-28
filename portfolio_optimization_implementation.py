'''1. Sets'''
tickers = ['AAPL', 'MSFT', 'GOOG','SAP', 'RGTI', 'TSLA', 'GLD','BTC']

# Initialize Pyomo model
model = pyo.ConcreteModel(name='Financial_Portfolio')
model.Assets = pyo.Set(initialize=tickers)

"""2. Parameters"""

'''LOCAL PARAMETERS'''
#1 returns of the set of assets
data = yf.download(tickers, start='2019-12-31', end='2024-12-31')#using API to obtain historical data of closing prices
closing_daily_prices = data["Close"] # Extract close prices to compute daily returns in percentages in the historical horizon
returns = closing_daily_prices.pct_change().dropna() # daily rate of change return
mean_daily_returns = returns.mean().values
mean_daily_returns=np.squeeze(mean_daily_returns)
mean_daily_returns=np.round(mean_daily_returns, 3)
#2 cost: closing price of today 02-01-2025
cost1=(yf.download(tickers, start='2025-01-02', end='2025-01-03'))['Close']
costs=np.squeeze(cost1)
costs=round(costs, 3)
#3 risk : variance  and covariance matrix
cov_matrix = returns.cov().values
#4 lminall lmaxall
Lminall = [0.05, 0.02, 0.03, 0.1, 0.15, 0.05, 0.01, 0.02]
Lmaxall = [0.8, 0.7, 0.9, 0.6, 0.85, 0.9, 0.7, 0.9]

'''GLOBAL PARAMETERS'''
B= 3000 #5 capital budget
Riskmax= 990  #6 riskmax
K=2 #min num of asset must be equal to the number of a
alpha= 0.9 #propenso al rischio
parameters = {
    'returns': {tickers[i]: mean_daily_returns[i] for i in range(len(tickers))},
    'costs': {tickers[i]: costs[i] for i in range(len(tickers))},
    'riskCovariance': {(tickers[i], tickers[j]): cov_matrix[i, j] for i in range(len(tickers)) for j in range(len(tickers))},
    'L_min_allocations': {tickers[i]: Lminall[i] for i in range(len(tickers))},
    'U_max_allocations': {tickers[i]: Lmaxall[i] for i in range(len(tickers))},
    'Budget': B,
    'Risk_Max': Riskmax,
    'Min_Assets': K,
    'Alpha': alpha,}

# local parameters to the model
model.returns = pyo.Param(model.Assets, initialize=parameters['returns'])
model.costs = pyo.Param(model.Assets, initialize=parameters['costs'])
model.L_min_allocations = pyo.Param(model.Assets, initialize=parameters['L_min_allocations'])
model.U_max_allocations = pyo.Param(model.Assets, initialize=parameters['U_max_allocations'])

# compute MArkovitz risk measure
model.riskCovariance = pyo.Param(model.Assets, model.Assets, initialize=parameters['riskCovariance'])
# Compute MAD risk measure
mad_risk = returns.sub(returns.mean()).abs().mean().values
model.madRisk = pyo.Param(model.Assets, initialize={tickers[i]: mad_risk[i] for i in range(len(tickers))})

#scalar parameters to the model
model.Budget = pyo.Param(initialize=parameters['Budget'])
model.Risk_Max = pyo.Param(initialize=parameters['Risk_Max'])
model.Min_Assets = pyo.Param(initialize=parameters['Min_Assets'])
model.Alpha = pyo.Param(initialize=parameters['Alpha'])
print('\n\n')

"""3. Variables"""

#Continuous variable with 0 , 1 bounds
model.x = pyo.Var(model.Assets, within=pyo.NonNegativeReals, bounds=(0, 1))

#Binary Variable
model.y=pyo.Var(model.Assets, within=pyo.Binary)

"""4. Constraints"""

def budget_constraint(model): #BUDGET
    return sum(model.costs[i] * model.x[i] for i in model.Assets) <= model.Budget
model.budget_constraint = pyo.Constraint(rule=budget_constraint)

def total_b(model): #CAPITAL
  return sum(model.x[i] for i in model.Assets)==1
model.total_b=pyo.Constraint(rule=total_b)

def diversificationk(model): #DIVERSIFICATION K
  return sum(model.y[i] for i in model.Assets) >= model.Min_Assets
model.diversificationk=pyo.Constraint(rule=diversificationk)
#model.diversification_constraint = pyo.Constraint(expr=sum(model.y[i] for i in model.Assets) >= model.Min_Assets)

def purchasing_constraint(model, i): #list of contraints
    return model.x[i] <= model.y[i]
model.purchasing_constraint = pyo.Constraint(model.Assets, rule=purchasing_constraint)
'''
model.purchasing_constraints = pyo.ConstraintList()
for i in model.Assets:
    model.purchasing_constraints.add(model.x[i] <= model.y[i])
'''
def min_allocation(model, i): #L Min
    return model.x[i] >= model.y[i]*model.L_min_allocations[i]  # Deve essere >= e non >
model.min_allocation_constr = pyo.Constraint(model.Assets, rule=min_allocation)

def max_allocation(model,i): #U Max
  return model.x[i] <= model.y[i]*model.U_max_allocations[i]
model.max_allocation_constr = pyo.Constraint(model.Assets, rule=max_allocation)
'''
model.investment_constraints = pyo.ConstraintList()
for i in model.Assets:
    model.investment_constraints.add(model.y[i] * model.L_min_allocations[i] <= model.x[i])
    model.investment_constraints.add(model.x[i] <= model.y[i] * model.U_max_allocations[i])
'''

# MAD RISK  constraints
model.risk_constraint_MAD = pyo.Constraint(expr=sum(model.x[i] * model.madRisk[i] for i in model.Assets) <= model.Risk_Max)

model.binaryConstraint= pyo.Constraint(expr=(model.y['BTC']+model.y['TSLA'])<=1)

"""5. Objective function"""

# Objective Function using MAD risk measure
model.obj = pyo.Objective(
    expr=(model.Alpha*sum(model.x[i] * model.returns[i] for i in model.Assets)) -
         ((1 - model.Alpha) * sum(model.x[i] * model.madRisk[i] for i in model.Assets)),
    sense=pyo.maximize)

"""6.1 Model solution"""

# Solve the model MAD
solver = pyo.SolverFactory('cbc')
results=solver.solve(model)

print(results.solver.termination_condition)
print(results.solver.status)

print("=== RESULTS WITH MAD RISK===")
for i in model.Assets:
  if model.y[i].value==0:
    continue
  print(f' X {i} : {pyo.value(model.x[i])}, Y : {pyo.value(model.y[i])}')

# Dictionary to store results
dict_of_results_MAD = {}
selected_assets = {i: model.y[i].value for i in model.Assets}
fractional_investments = {i: model.x[i].value for i in model.Assets}
portfolio_return = sum(model.x[i] * model.returns[i] for i in model.Assets)
portfolio_risk = sum(model.x[i] * model.madRisk[i] for i in model.Assets)

from pyomo.environ import value

# Extracting and printing the values of the objective function, portfolio return, and portfolio risk
objective_value = value(model.obj)
portfolio_return_value = value(portfolio_return)
portfolio_risk_value = value(portfolio_risk)
dict_of_results_MAD = {
        "objective_value": objective_value,
        "portfolio_return": portfolio_return_value,
        "portfolio_risk": portfolio_risk_value,
        "selected_assets_Y": selected_assets,
        "x_fractional_investments_X": fractional_investments}
pprint(dict_of_results_MAD)

"""6.2 Analysis"""

# Assuming model has been defined and solver is set up

alpha_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Values for alpha
results = []  # To store the results for each alpha

# Loop over each alpha value
for alpha in alpha_values:
    # Update the model's objective function based on the current value of alpha
    model.obj = pyo.Objective(
        expr=(alpha * sum(model.x[i] * model.returns[i] for i in model.Assets)) -
             ((1 - alpha) * sum(model.x[i] * model.madRisk[i] for i in model.Assets)),
        sense=pyo.maximize
    )

    # Solve the model
    solver = SolverFactory('cbc')
    solver.solve(model, tee=False)

    # Collect the results: Total return, total risk, and objective value
    total_return = sum(model.x[i].value * model.returns[i] for i in model.Assets)
    total_risk = sum(model.x[i].value * model.madRisk[i] for i in model.Assets)
    total_obj_value = value(model.obj)  # Get the objective value

    # Store the results
    results.append({
        'Alpha': alpha,
        'Total Return': total_return,
        'Total Risk (MAD)': total_risk,
        'Objective Value': total_obj_value
    })
print('\n\n\n')
print('=====================Results=====================')
pprint(results)

""" 6.3 - Model solution : with Markovitz"""

#Continuous variable with 0 , 1 bounds
model.x = pyo.Var(model.Assets, within=pyo.NonNegativeReals, bounds=(0, 1))
#Binary Variable
model.y=pyo.Var(model.Assets, within=pyo.Binary)
#for markovitz model
model.z=pyo.Var(model.Assets, model.Assets, within=pyo.NonNegativeReals, bounds=(0, 1))

#CONSTRAINTS

def budget_constraint(model): #BUDGET
    return sum(model.costs[i] * model.x[i] for i in model.Assets) <= model.Budget
model.budget_constraint = pyo.Constraint(rule=budget_constraint)

def total_b(model): #CAPITAL
  return sum(model.x[i] for i in model.Assets)==1
model.total_b=pyo.Constraint(rule=total_b)

def diversificationk(model): #DIVERSIFICATION K
  return sum(model.y[i] for i in model.Assets) >= model.Min_Assets
model.diversificationk=pyo.Constraint(rule=diversificationk)
#model.diversification_constraint = pyo.Constraint(expr=sum(model.y[i] for i in model.Assets) >= model.Min_Assets)

def purchasing_constraint(model, i): #list of contraints
    return model.x[i] <= model.y[i]
model.purchasing_constraint = pyo.Constraint(model.Assets, rule=purchasing_constraint)
'''
model.purchasing_constraints = pyo.ConstraintList()
for i in model.Assets:
    model.purchasing_constraints.add(model.x[i] <= model.y[i])
'''
def min_allocation(model, i): #L Min
    return model.x[i] >= model.y[i]*model.L_min_allocations[i]  # Deve essere >= e non >
model.min_allocation_constr = pyo.Constraint(model.Assets, rule=min_allocation)

def max_allocation(model,i): #U Max
  return model.x[i] <= model.y[i]*model.U_max_allocations[i]
model.max_allocation_constr = pyo.Constraint(model.Assets, rule=max_allocation)
'''
model.investment_constraints = pyo.ConstraintList()
for i in model.Assets:
    model.investment_constraints.add(model.y[i] * model.L_min_allocations[i] <= model.x[i])
    model.investment_constraints.add(model.x[i] <= model.y[i] * model.U_max_allocations[i])
'''

#MARKOVITZ CONSTRAINT
def Mark_risk_contraint (model):
    return sum(model.z[i, j]*model.riskCovariance[i,j] for i in model.Assets for j in model.Assets) <= model.Risk_Max
model.risk_constraint_Mark = pyo.Constraint(rule=Mark_risk_contraint)

#MODELLO 1 Markovitz with linearization of xixy
#  variabiles' constraints
def mccormick1(model, i, j):
    return model.z[i, j] >= model.x[i] + model.x[j] - 1

def mccormick2(model, i, j):
    return model.z[i, j] <= model.x[i]

def mccormick3(model, i, j):
    return model.z[i, j] <= model.x[j]

def mccormick4(model, i, j):
    return model.z[i, j] >= 0

model.mccormick1 = pyo.Constraint(model.Assets, model.Assets, rule=mccormick1)
model.mccormick2 = pyo.Constraint(model.Assets, model.Assets, rule=mccormick2)
model.mccormick3 = pyo.Constraint(model.Assets, model.Assets, rule=mccormick3)
model.mccormick4 = pyo.Constraint(model.Assets, model.Assets, rule=mccormick4)


#OBJECTIVE MARKOVITZ
def total_return(model):
  tot_return=sum(model.returns[i] * model.x[i]  for i in model.Assets)
  tot_risk=sum(model.z[i,j]*model.riskCovariance[i,j] for i in model.Assets for j in model.Assets)
  return tot_return*model.Alpha - (1-model.Alpha)*tot_risk
model.obj = pyo.Objective(rule= total_return, sense=pyo.maximize)


print("=== RESULTS ===")
#model.display()
solver = pyo.SolverFactory('cbc')
results=solver.solve(model, tee=False)
print(results.solver.termination_condition)
print(results.solver.status)

for i in model.Assets:
  if model.y[i].value==0:
    continue
  print(f' X {i} : {pyo.value(model.x[i])}, Y : {pyo.value(model.y[i])}')

for i in model.Assets:
  for j in model.Assets:
    if model.z[i,j].value==0:
      continue
    print(f"z {i,j}: {model.z[i,j].value}")

for i in model.Assets:
  for j in model.Assets:
    if model.x[i].value*model.x[j].value==0:
      continue
    print(f"x{i, j}: {model.x[i].value, model.x[j].value, model.x[i].value*model.x[j].value}")

print('\n\n')
#print('dev std con z:', sqrt(0.75*0.015936055903302365))


dict_of_results_Mark = {}
selected_assets_Mark = {i: model.y[i].value for i in model.Assets}
fractional_investments_Mark = {i: model.x[i].value for i in model.Assets}
portfolio_return_Mark = sum(fractional_investments[i] * parameters['returns'][i] for i in tickers)
portfolio_risk_Mark = sum(model.x[i] * model.madRisk[i] for i in model.Assets)
dict_of_results_Mark = {
        "objective_value": pyo.value(model.obj),
        "portfolio_return": portfolio_return_Mark,
        "portfolio_risk": value(portfolio_risk_Mark),
        "selected_assets_Y": selected_assets_Mark,
        "x_fractional_investments_X": fractional_investments_Mark}
pprint(dict_of_results_Mark)
