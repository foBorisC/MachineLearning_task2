import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import tree
from scripts.data_processing import preprocess_data_with_cyclical

if __name__ == "__main__":

    whole_df = preprocess_data_with_cyclical()

    #Separate features and target, not need to normalize for tree based models
    X = whole_df.drop('count', axis=1)
    Y = whole_df['count']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    #SVR je CITLIVÝ na škálu dát - MUSÍTE normalizovať!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Základný SVR
    svr_model = SVR(
        kernel='rbf',  # Typ jadra
        C=69.0,  # Regularizácia
        epsilon=5,  # Šírka pásu tolerancie
        gamma='scale'  # Šírka RBF jadra
    )

    #Trénovanie
    svr_model.fit(X_train_scaled, y_train)

    #Predikcia a vyhodnotenie
    y_pred = svr_model.predict(X_test_scaled)
    y_pred_train = svr_model.predict(X_train_scaled)

    mse_test = mean_squared_error(y_test, y_pred)
    rmse_test = np.sqrt(mse_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)

    print("Support Vector Regressor Results:")
    print(f"MSE (test): {mse_test:.2f}")
    print(f"RMSE (test): {rmse_test:.2f}")
    print(f"MSE (train): {mse_train:.2f}")
    print(f"RMSE (train): {rmse_train:.2f}")
    print(f"R2 train: {r2_score(y_train, y_pred_train):.3f}")
    print(f"R2 test: {r2_score(y_test, y_pred):.3f}")


    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predpovedané hodnoty')
    plt.ylabel('Reziduá (skutočné - predpovedané)')
    plt.title('Reziduá vs. Predpovedané hodnoty SVR')
    plt.grid(alpha=0.3)
    plt.show()