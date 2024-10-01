# Titanic-Survival-Prediction-using-Machine-Learning

This repository contains a machine learning project aimed at predicting the survival of passengers aboard the Titanic using the famous Titanic dataset from Kaggle. The project utilizes various machine learning models to analyze and predict survival based on a combination of features.

## Dataset

The dataset used for this project is sourced from [Kaggle's Titanic - Machine Learning from Disaster competition](https://www.kaggle.com/c/titanic/data). It contains data about passengers aboard the Titanic, including:

- PassengerId
- Pclass (Ticket class)
- Name
- Sex
- Age
- SibSp (# of siblings / spouses aboard the Titanic)
- Parch (# of parents / children aboard the Titanic)
- Ticket
- Fare
- Cabin
- Embarked (Port of Embarkation)

### Additional Columns Added:
- **FamilySize**: Combination of SibSp and Parch to create a new feature representing the total number of family members aboard.
- **Title**: Extracted from the passenger's name to group passengers based on their title (e.g., Mr., Mrs., Miss.).
- **Fare per Person**: Fare divided by FamilySize to normalize ticket fare.
  
## Machine Learning Models Used
In this project, multiple machine learning algorithms have been implemented to predict the likelihood of survival. These models include:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**

The models are trained and evaluated based on the accuracy. 

## Installation and Requirements

### Prerequisites

To run this project locally, you will need to install the following dependencies:

- Python 3.x
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

You can install the necessary packages by running the following command:

```bash
pip install -r requirements.txt
```

### Instructions to Run

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
```

2. Navigate to the project directory:

```bash
cd titanic-survival-prediction
```

3. Open the Jupyter notebook to view and run the code:

```bash
jupyter notebook titanic-survival-prediction.ipynb
```

4. Download the dataset from [Kaggle's Titanic Dataset](https://www.kaggle.com/c/titanic/data) and place it in the `data/` folder.

5. Follow the steps in the notebook to preprocess the data, train the models, and evaluate their performance.

## Results

After training and testing the models, the best-performing model can be selected based on its performance metrics. The results for each model will be displayed in the notebook.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request to contribute to this project.

