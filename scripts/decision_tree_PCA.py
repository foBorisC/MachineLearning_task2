import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scripts.data_processing import preprocess_data_for_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":

    whole_df = preprocess_data_for_tree()
    df = whole_df.copy()

    # Priprava dát
    X = df.drop('count', axis=1)
    y = df['count']
    feature_names = X.columns

    # Normalizácia
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    print("=" * 60)
    print("TRÉNOVANIE S PCA TRANSFORMOVANÝMI DÁTAMI")
    print("=" * 60)

    # PCA pre rôzne úrovne variability
    variance_levels = [0.70, 0.80, 0.90]

    for variance in variance_levels:
        print(f"\n--- PCA s {variance * 100}% variability ---")

        # PCA transformácia
        pca = PCA(n_components=variance)
        X_pca = pca.fit_transform(X_normalized)

        print(f"Počet PCA komponentov: {X_pca.shape[1]}")
        print(f"Skutočne zachovaná variabilita: {sum(pca.explained_variance_ratio_) * 100:.1f}%")

        # Rozdelenie dát
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.2, random_state=42
        )

        # Trénovanie Decision Tree
        dtree = DecisionTreeRegressor(max_depth=10, random_state=42)
        dtree.fit(X_train, y_train)

        # Evaluate on TRAIN and TEST data
        y_pred_train = dtree.predict(X_train)
        y_pred = dtree.predict(X_test)

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

        # Reziduálny graf
        residuals = y_test - y_pred

        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predpovedané hodnoty')
        plt.ylabel('Reziduá (skutočné - predpovedané)')
        plt.title(f'Reziduá vs. Predpovedané hodnoty (PCA {variance * 100}% variability)')
        plt.show()

    # Porovnanie so všetkými príznakmi (bez PCA)
    print(f"\n--- BEZ PCA (všetky {X.shape[1]} príznakov) ---")

    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    dtree_all = DecisionTreeRegressor(max_depth=10, random_state=42)
    dtree_all.fit(X_train_all, y_train)

    y_pred_train_all = dtree_all.predict(X_train_all)
    y_pred_all = dtree_all.predict(X_test_all)

    print(f"R2 train: {r2_score(y_train, y_pred_train_all):.3f}")
    print(f"R2 test: {r2_score(y_test, y_pred_all):.3f}")
    print(f"RMSE test: {np.sqrt(mean_squared_error(y_test, y_pred_all)):.2f}")