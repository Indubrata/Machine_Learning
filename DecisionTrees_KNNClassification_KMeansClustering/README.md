# Star Classification: KNN, K-Means & Decision Trees

This repository contains a Jupyter Notebook demonstrating the application of three fundamental machine learning algorithms on a star classification dataset. The goal is to accurately classify celestial objects (like stars, galaxies, and quasars) based on their spectral characteristics and to explore their groupings.

## Project Structure

- `DecisionTrees_KNNClassification_KMeansClustering.ipynb`: The main Jupyter Notebook containing all the code, visualizations, and detailed explanations of the algorithms.
- `star_classification.csv`: The dataset used for training and evaluating the models.

## Algorithms Explored

1. **K-Nearest Neighbors (KNN)**: A supervised learning method used here for classification. The notebook demonstrates how it predicts the class of an object based on the closest data points in the feature space using Euclidean distance.
2. **K-Means Clustering**: An unsupervised learning method used to discover inherent groupings within the data. The notebook utilizes the Elbow Method to find the optimal number of clusters and visualizes the resulting centroids.
3. **Decision Trees**: A supervised learning algorithm that builds a flowchart-like structure to classify the data based on feature splits (using criteria like Information Gain/Entropy and Gini Index). The notebook includes a visual plot of the tree.

## Prerequisites

To run the notebook locally, you need Python installed along with the following libraries:

- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## How to Run

1. Clone this repository or download the folder.
2. Ensure the `star_classification.csv` file is in the same directory as the notebook.
3. Open a terminal or command prompt in this directory.
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Open `DecisionTrees_KNNClassification_KMeansClustering.ipynb` and run the cells sequentially.
