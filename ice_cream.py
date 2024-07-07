from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

current_directory = os.getcwd()
df = pd.DataFrame(pd.read_csv(os.path.join(current_directory, 'ice_cream\\Ice Cream.csv')))

# splitting
X = df['Temperature'].values.reshape(-1, 1)  # 1D -> 2D array
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_cross_validated = RandomForestRegressor(n_estimators=100, random_state=42)

# cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_mse = cross_val_score(model_cross_validated, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
cross_val_r2 = cross_val_score(model_cross_validated, X_train_scaled, y_train, cv=kf, scoring='r2')

model_cross_validated.fit(X_train_scaled, y_train)
y_pred_cross_validated = model_cross_validated.predict(X_test_scaled)

# metrics
mean_cross_val_mse = -cross_val_mse.mean()
std_cross_val_mse = cross_val_mse.std()
mean_cross_val_r2 = cross_val_r2.mean()
std_cross_val_r2 = cross_val_r2.std()

mse_rf = mean_squared_error(y_test, y_pred_cross_validated)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_cross_validated)

print(f'Cross-Validation MSE: {mean_cross_val_mse:.4f} ± {std_cross_val_mse:.4f}')
print(f'Cross-Validation R²: {mean_cross_val_r2:.4f} ± {std_cross_val_r2:.4f}')
print(f'Test MSE: {mse_rf:.4f}')
print(f'Test RMSE: {rmse_rf:.4f}')
print(f'Test R²: {r2_rf:.4f}')

plt.scatter(y_test, y_pred_cross_validated, color='r', alpha=0.2, edgecolors='b', label='Test Data')
plt.plot(y_test, y_pred_cross_validated, linestyle='--', color='b', alpha=0.4, label='Test Prediction Trend')
plt.legend()
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')
plt.title('Actual vs. Predicted Revenue')
plt.show()