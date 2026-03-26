# Variational Quantum Circuit (VQC) for Lottery Prediction
# Quantum Regression Model with Qiskit

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

from qiskit_machine_learning.utils import algorithm_globals
import random

# ================= SEED PARAMETERS =================
SEED = 39
random.seed(SEED)
np.random.seed(SEED)
algorithm_globals.random_seed = SEED
# ==================================================


# Use the existing dataframe
df_raw = pd.read_csv('/data/loto7hh_4586_k24.csv')
# 4586 historical draws of Lotto 7/39 (Serbia)

_MIN_POS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
_MAX_POS = np.array([33, 34, 35, 36, 37, 38, 39], dtype=int)


def quantum_regression_predict(df):
    df = df.copy()
    cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6', 'Num7']
    
    # Prepare lag features
    for col in cols:
        df[f'{col}_lag'] = df[col].shift(1)
    
    # v2: manji prozor da optimizacija ne "stoji" (brže krene).
    df_model = df.dropna().tail(1200) # 200
    
    predictions = {}
    
    # Scaling
    scaler_x = MinMaxScaler(feature_range=(0, np.pi))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    
    # Quantum Circuit Setup
    num_qubits = 1
    x_param = ParameterVector('x', 1)
    theta_param = ParameterVector('theta', 2)
    
    # Feature Map + Ansatz
    qc = QuantumCircuit(num_qubits)
    qc.ry(x_param[0], 0) # Encoding
    qc.ry(theta_param[0], 0) # Trainable
    qc.rz(theta_param[1], 0) # Trainable
    
    observable = SparsePauliOp('Z')
    estimator = StatevectorEstimator()
    
    def eval_pred(x_val, params):
        pub = (qc, observable, [x_val, params[0], params[1]])
        job = estimator.run([pub])
        result = job.result()[0]
        evs = result.data.evs # Expectation value
        return float(np.real(np.asarray(evs).reshape(-1)[0]))

    def cost_function(params, X, y):
        # params: current weights [theta0, theta1]
        # X: scaled inputs
        # y: scaled targets
        mse = 0.0
        for i in range(len(X)):
            prediction = eval_pred(X[i][0], params)
            mse += (prediction - y[i])**2
            
        return mse / len(X)

    for idx, col in enumerate(cols):
        print(f"\n[VQC_v2] Treniranje za poziciju {idx+1} ({col})...")
        X = df_model[[f'{col}_lag']].values
        y = df_model[col].values.reshape(-1, 1)
        
        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y).flatten()

        # v2: trošak je skup (estimator poziva se u svakoj evaluaciji cost funkcije),
        # pa optimizaciju radimo na zadnjem delu prozora radi brzine.
        n_sub = min(80, len(X_scaled))
        X_opt = X_scaled[-n_sub:]
        y_opt = y_scaled[-n_sub:]
        
        # Initial weights
        # v2: robustniji fit sa više startova
        best_x = None
        best_cost = float("inf")
        for restart in range(2):
            init_params = np.random.uniform(0, 2*np.pi, 2)
            print(f"[VQC_v2] restart {restart+1}/2 init={init_params}")
            
            # Optimize
            res = minimize(
                cost_function,
                init_params,
                args=(X_opt, y_opt),
                method='COBYLA',
                options={'maxiter': 40, 'rhobeg': 0.25}
            )
            c = float(res.fun)
            if c < best_cost:
                best_cost = c
                best_x = res.x
            print(f"[VQC_v2] restart {restart+1}/2 cost={c:.6f} best_cost={best_cost:.6f}")
        
        # Predict next
        last_val = np.array([[df[col].iloc[-1]]])
        last_val_scaled = scaler_x.transform(last_val)
        
        final_pred_scaled = eval_pred(last_val_scaled[0][0], best_x)
        
        # Inverse scale
        pred_final = scaler_y.inverse_transform(np.array([[final_pred_scaled]]))
        
        # Bound to reasonable lottery numbers
        lo, hi = int(_MIN_POS[idx]), int(_MAX_POS[idx])
        predictions[col] = int(round(np.clip(pred_final[0][0], lo, hi)))
        
    return predictions

# Run the Quantum Prediction
quantum_results = quantum_regression_predict(df_raw)

# Format for display
quantum_pred_df = pd.DataFrame([quantum_results])
# quantum_pred_df.index = ['Quantum Regression Prediction (VQC)']

print()
print("Lottery prediction generated using a Variational Quantum Circuit (VQC) for regression.")
print()


print()
print("Variational Quantum Circuit (VQC) Results:")
print(quantum_pred_df.to_string(index=True))
print()
"""
[VQC_v2] Treniranje za poziciju 1 (Num1)...
[VQC_v2] restart 1/2 init=[3.43620591 5.01334741]
[VQC_v2] restart 1/2 cost=0.160972 best_cost=0.160972
[VQC_v2] restart 2/2 init=[5.15473704 0.76686193]
[VQC_v2] restart 2/2 cost=0.160972 best_cost=0.160972

[VQC_v2] Treniranje za poziciju 2 (Num2)...
[VQC_v2] restart 1/2 init=[3.78249016 3.30190551]
[VQC_v2] restart 1/2 cost=0.507906 best_cost=0.507906
[VQC_v2] restart 2/2 init=[2.91482248 2.96218097]
[VQC_v2] restart 2/2 cost=0.507906 best_cost=0.507906

[VQC_v2] Treniranje za poziciju 3 (Num3)...
[VQC_v2] restart 1/2 init=[3.97545204 5.81611769]
[VQC_v2] restart 1/2 cost=0.449833 best_cost=0.449833
[VQC_v2] restart 2/2 init=[5.12395594 5.93410254]
[VQC_v2] restart 2/2 cost=0.449833 best_cost=0.449833

[VQC_v2] Treniranje za poziciju 4 (Num4)...
[VQC_v2] restart 1/2 init=[5.77792385 2.60645876]
[VQC_v2] restart 1/2 cost=0.527483 best_cost=0.527483
[VQC_v2] restart 2/2 init=[5.18872959 5.94552768]
[VQC_v2] restart 2/2 cost=0.463474 best_cost=0.463474

[VQC_v2] Treniranje za poziciju 5 (Num5)...
[VQC_v2] restart 1/2 init=[3.94612973 1.58505785]
[VQC_v2] restart 1/2 cost=0.502150 best_cost=0.502150
[VQC_v2] restart 2/2 init=[3.72185049 5.40607658]
[VQC_v2] restart 2/2 cost=0.383721 best_cost=0.383721

[VQC_v2] Treniranje za poziciju 6 (Num6)...
[VQC_v2] restart 1/2 init=[0.30443561 5.37085767]
[VQC_v2] restart 1/2 cost=0.297336 best_cost=0.297336
[VQC_v2] restart 2/2 init=[4.52523813 1.85965595]
[VQC_v2] restart 2/2 cost=0.297336 best_cost=0.297336

[VQC_v2] Treniranje za poziciju 7 (Num7)...
[VQC_v2] restart 1/2 init=[3.04989881 5.3047394 ]
[VQC_v2] restart 1/2 cost=0.117871 best_cost=0.117871
[VQC_v2] restart 2/2 init=[3.7855635  0.98737631]
[VQC_v2] restart 2/2 cost=0.116444 best_cost=0.116444

Lottery prediction generated using a Variational Quantum Circuit (VQC) for regression.


Variational Quantum Circuit (VQC) Results:
   Num1  Num2  Num3  Num4  Num5  Num6  Num7
0     7     9     3     x    y    33    z
"""



"""
df.copy() da se izbegne mutacija ulaza,
trening prozor je tail(1200),
optimizacija koristi poslednjih ~80 uzoraka,
robustniji fit: COBYLA sa 2 restarta, maxiter=40, rhobeg=0.25,
stabilno čitanje evs kao float,
clip po pozicijama za 7/39 (1..33, 2..34, ..., 7..39).


prozor za trening: tail(1200)
broj restarta: 4 -> 2
COBYLA maxiter: 120 -> 40
optimizacija cost-a sad koristi poslednjih ~80 uzoraka (X_opt/y_opt) umesto celog prozora
dodati print:
na početku svake pozicije (Num1..Num7)
pre i posle svakog restart-a (init, cost, best_cost)
"""
