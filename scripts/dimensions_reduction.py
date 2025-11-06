import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scripts.data_processing import preprocess_data_for_tree
import matplotlib.pyplot as plt
import pandas as pd

'''
    Redukcia dimenzie.
    Časť 1: Ručne vybrané príznaky do 3D grafu.
    Časť 2: PCA redukcia do 3D grafu.
'''

if __name__ == "__main__":

    '''
        
        1. Časť
    
    '''
    whole_df = preprocess_data_for_tree()
    df = whole_df.copy()

    # Vyberieme 3 zmysluplné príznaky
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    #3D scatter plot
    scatter = ax.scatter(
        df['weekday'],  # X-axis - day of the week
        df['temperature'],  # Y-axis - temperature
        df['month'],  # Z-axis - month of the year
        c=df['count'],  # Color by bike count
        cmap='viridis',
        alpha=0.6
    )

    ax.set_xlabel('Deň v týždni')
    ax.set_ylabel('Teplota (°C)')
    ax.set_zlabel('Mesiac v roku')
    plt.colorbar(scatter, label='Počet bicyklov')
    plt.title('3D graf: Ručne vybrané príznaky')
    plt.show()
    '''
    
        2. Časť
    
    '''
    #Normalisation
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(df.drop('count', axis=1))

    #Reduction with PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_normalized)

    #Visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=df['count'],
        cmap='viridis',
        alpha=0.6
    )

    ax.set_xlabel('PCA 1 (najviac variabilita)')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    plt.colorbar(scatter, label='Počet bicyklov')
    plt.title('3D graf: PCA redukcia dimenzie')
    plt.show()

    # Koľko variability zachovali prvé 3 zložky
    print(f"PCA zachováva {sum(pca.explained_variance_ratio_) * 100:.1f}% variability")
    # Vytvoríte DataFrame s komponentami
    pca_components_df = pd.DataFrame(
        pca.components_.T,  # Komponenty transponované
        columns=['PCA1', 'PCA2', 'PCA3'],  # Názvy stĺpcov
        index=df.drop('count', axis=1).columns  # Názvy pôvodných príznakov
    )

    # Vypíšte čo obsahuje PCA1 - ZORADENÉ OD NAJVÄČŠIEHO
    print("=== ČO OBSAHUJE PCA1 ===")
    print(pca_components_df['PCA1'].sort_values(ascending=False))

    # 1. Korelačná matica z PÔVODNÝCH dát (pred normalizáciou)
    corr_matrix = df.corr()

    # 2. Vizualizácia
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                square=True)
    plt.title('Korelačná matica - pôvodné dáta')
    plt.tight_layout()
    plt.show()

    # 3. Výber príznakov podľa korelácie s cieľovou premennou
    target_correlation = corr_matrix['count'].sort_values(ascending=False)

    print("=== KORELÁCIA S COUNT (počet bicyklov) ===")
    print(target_correlation)