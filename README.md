# iris_KNN_Classification.
KNN Classification_PCA.
## Project Description

This project uses the Iris dataset to train a K-Nearest Neighbors (KNN) classifier to predict the species of Iris flowers. The dataset has four features: SepalLengthCm, SepalWidthCm, PetalLengthCm, and PetalWidthCm, which are used to classify the species of Iris flowers into three categories:

Iris-setosa

Iris-versicolor

Iris-virginica

The features are first normalized and then reduced to 2D using Principal Component Analysis (PCA) for visualization purposes. The classifier is evaluated by plotting the decision boundaries and showing how the KNN model classifies different regions in the PCA-reduced space.

## Requirements

To run this project, you'll need the following Python libraries:

numpy

matplotlib

seaborn

scikit-learn

pandas

You can install these libraries using pip:
pip install numpy matplotlib seaborn scikit-learn pandas

## Dataset

The Iris dataset is a well-known dataset for machine learning tasks, available through scikit-learn or from sources like Kaggle.

Features: Sepal Length, Sepal Width, Petal Length, Petal Width

Target: Species of Iris flower (Setosa, Versicolor, Virginica)

## Steps to Run the Code
1. Load the Data

The Iris dataset is loaded into a Pandas DataFrame. If using a CSV file, simply replace the dataset loading part with:

df = pd.read_csv('path_to_iris.csv')

2. Data Preprocessing

Normalization: All features are scaled using StandardScaler to ensure that each feature contributes equally to the model.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

3. Apply PCA for Dimensionality Reduction

PCA reduces the original feature space (4 features) to 2D for visualization. The first two principal components are used for plotting the decision boundary.

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

4. Train the KNN Classifier

K-Nearest Neighbors (KNN) is used to classify the flowers based on the PCA-reduced data. The optimal number of neighbors (k) is determined through experimentation.

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_pca, y_train_encoded)

5. Visualize Decision Boundaries

The decision boundaries of the KNN classifier are plotted on the PCA-reduced space. The model's predictions are shown as colored regions, with scatter plots of the training data points overlaid on top.

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_encoded, edgecolors='k', s=100, cmap=plt.cm.RdYlBu)

6. Evaluate the Model

You can evaluate the classifier using cross-validation, accuracy score, or confusion matrix for performance evaluation.

from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test_pca)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

## Example Output

After running the code, you will see a decision boundary plot with the Iris dataset in the PCA-reduced 2D space. The plot will show:

Colored decision regions based on the KNN classifier's predictions.

Scatter points representing the training samples, color-coded by their true class.

## Conclusion

This project demonstrates how to apply Principal Component Analysis (PCA) for dimensionality reduction and K-Nearest Neighbors (KNN) for classification on the Iris dataset. The decision boundary visualization helps in understanding how the classifier makes decisions and shows the separation between different classes.

Feel free to modify the k value for the KNN classifier and experiment with different numbers of PCA components for dimensionality reduction.
