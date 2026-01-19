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
df_raw = pd.read_csv('/data/loto7hh_4548_k5.csv')
# 4548 historical draws of Lotto 7/39 (Serbia)




def quantum_regression_predict(df):
    cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6', 'Num7']
    
    # Prepare lag features
    for col in cols:
        df[f'{col}_lag'] = df[col].shift(1)
    
    # Use a small subset 20 for training speed in the kernel
    df_model = df.dropna().tail(4548)
    
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
    
    def cost_function(params, X, y):
        # params: current weights [theta0, theta1]
        # X: scaled inputs
        # y: scaled targets
        
        mse = 0
        for i in range(len(X)):
            # Bind parameters
            # Note: StatevectorEstimator.run takes pubs (circuit, observable, parameter_values)
            pub = (qc, observable, [X[i][0], params[0], params[1]])
            job = estimator.run([pub])
            result = job.result()[0]
            prediction = result.data.evs # Expectation value
            mse += (prediction - y[i])**2
            
        return mse / len(X)

    for col in cols:
        X = df_model[[f'{col}_lag']].values
        y = df_model[col].values.reshape(-1, 1)
        
        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y).flatten()
        
        # Initial weights
        init_params = np.random.rand(2)
        
        # Optimize
        res = minimize(cost_function, init_params, args=(X_scaled, y_scaled), method='COBYLA', options={'maxiter': 10})
        
        # Predict next
        last_val = np.array([[df[col].iloc[-1]]])
        last_val_scaled = scaler_x.transform(last_val)
        
        final_pub = (qc, observable, [last_val_scaled[0][0], res.x[0], res.x[1]])
        final_job = estimator.run([final_pub])
        final_pred_scaled = final_job.result()[0].data.evs
        
        # Inverse scale
        pred_final = scaler_y.inverse_transform(np.array([[final_pred_scaled]]))
        
        # Bound to reasonable lottery numbers
        predictions[col] = max(1, int(round(pred_final[0][0])))
        
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
Variational Quantum Circuit (VQC) Results:
   Num1  Num2  Num3  Num4  Num5  Num6  Num7
0     4    10     x     y     z    28    36
"""
