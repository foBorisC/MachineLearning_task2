import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
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

    print("=" * 60)
    print("TRÉNOVANIE S PCA TRANSFORMOVANÝMI DÁTAMI")
    print("=" * 60)

    # PCA pre rôzne úrovne variability
    variance_levels = [0.70, 0.80, 0.90]

    for variance in variance_levels:
        print(f"\n--- PCA s {variance * 100}% variability ---")

        # ✅ SPRÁVNE: PCA LEN na features (BEZ count)
        pca = PCA(n_components=variance)
        X_pca = pca.fit_transform(X)  # Iba features, žiadny count!

        print(f"Počet PCA komponentov: {X_pca.shape[1]}")
        print(f"Skutočne zachovaná variabilita: {sum(pca.explained_variance_ratio_) * 100:.1f}%")

        # Rozdelenie dát
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.2, random_state=42
        )

        # Trénovanie Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )

        # Trénovanie
        rf_model.fit(X_train, y_train)

        # Predikcia a vyhodnotenie
        y_pred = rf_model.predict(X_test)
        y_pred_train = rf_model.predict(X_train)

        mse_test = mean_squared_error(y_test, y_pred)
        rmse_test = np.sqrt(mse_test)

        mse_train = mean_squared_error(y_train, y_pred_train)
        rmse_train = np.sqrt(mse_train)

        print("Random Forest Results:")
        print(f"R2 train: {r2_score(y_train, y_pred_train):.3f}")
        print(f"R2 test: {r2_score(y_test, y_pred):.3f}")
        print(f"RMSE test: {rmse_test:.2f}")
        print(f"RMSE train: {rmse_train:.2f}")

        # Reziduálny graf
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predpovedané hodnoty')
        plt.ylabel('Reziduá (skutočné - predpovedané)')
        plt.title(f'Reziduá vs. Predpovedané (PCA {variance * 100}% variability)')
        plt.show()

    # Porovnanie so všetkými príznakmi (bez PCA)
    print(f"\n--- BEZ PCA (všetky {X.shape[1]} príznakov) ---")

    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Použite rovnaký Random Forest pre konzistentné porovnanie
    rf_all = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf_all.fit(X_train_all, y_train)

    y_pred_all = rf_all.predict(X_test_all)
    print(f"R2 test (bez PCA): {r2_score(y_test, y_pred_all):.3f}")
    print(f"RMSE test (bez PCA): {np.sqrt(mean_squared_error(y_test, y_pred_all)):.2f}")