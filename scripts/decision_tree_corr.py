import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from scripts.data_processing import preprocess_data_for_tree

if __name__ == "__main__":

    whole_df = preprocess_data_for_tree()

    #Separate features and target, not need to normalize for tree based models
    selected_features = ['hour','humidity','temperature']
    X_corr = whole_df[selected_features]
    Y = whole_df['count']

    X_train, X_test, y_train, y_test = train_test_split(X_corr, Y, test_size=0.3, random_state=42)

    rf_model = RandomForestRegressor(
        n_estimators=100,  # Počet stromov v lese
        max_depth=15,  # Maximálna hĺbka každého stromu
        min_samples_split=40,  # Min. vzoriek na rozdelenie uzlu
        min_samples_leaf=20,  # Min. vzoriek v liste
        random_state=42,  # Pre reprodukovateľnosť
        n_jobs=-1  # Použiť všetky jadra procesora
    )

    # Trénovanie
    rf_model.fit(X_train, y_train)

    #Evaluate on TEST data
    y_pred_train = rf_model.predict(X_train)
    y_pred = rf_model.predict(X_test)

    mse_test = mean_squared_error(y_test, y_pred)
    rmse_test = np.sqrt(mse_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)

    print("Decision Tree Regressor Results:")
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
    plt.title('Reziduá vs. Predpovedané hodnoty Decision Tree')
    plt.show()