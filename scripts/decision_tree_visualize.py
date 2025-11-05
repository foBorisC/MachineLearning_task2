import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import tree
from scripts.data_processing import preprocess_data_for_tree

if __name__ == "__main__":

    whole_df = preprocess_data_for_tree()

    #Separate features and target, not need to normalize for tree based models
    X = whole_df.drop('count', axis=1)
    Y = whole_df['count']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    dtree = DecisionTreeRegressor(max_depth=3, random_state=49)
    dtree = dtree.fit(X_train, y_train)

    #Evaluate on TEST data
    y_pred = dtree.predict(X_test)
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R2: {r2_score(y_test, y_pred):.2f}")

    #Visualize
    plt.figure(figsize=(20, 12))
    plot_tree(dtree,
              feature_names=X.columns,
              filled=True,
              fontsize=10,
              rounded=True)
    plt.tight_layout()
    plt.show()
