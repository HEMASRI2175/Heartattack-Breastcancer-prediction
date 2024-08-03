

## Inspiration
The inspiration behind this project stemmed from the need for accessible and accurate health risk assessments. With heart disease and breast cancer being leading causes of mortality worldwide, we aimed to create a tool that empowers individuals with early predictions, helping them take proactive steps towards their health. Leveraging machine learning technology, we aspired to bring advanced health diagnostics to everyone's fingertips.

## What it does
The "Heart Attack & Breast Cancer Prediction" app utilizes machine learning algorithms to predict the risk of heart attacks and breast cancer. Users input specific attributes such as age, sex, heart rate, and blood sugar levels, and the app provides a risk assessment (High Risk or Low Risk). This tool not only predicts health risks but also educates users on how different factors impact their health.
### Overview of the "ML in Healthcare" Web App

The "ML in Healthcare" web app is an interactive platform developed using Python and Streamlit to predict the risk of heart attacks or breast cancer. It leverages machine learning (ML) algorithms to build robust and accurate models based on user-specific attributes like age, sex, heart rate, blood sugar levels, and more. Users can interact with the app to both build and test models as well as make predictions based on their input data.


### Sections of the Application

#### 1. Model Building
In the Model Building section, the app builds and trains seven different machine learning models using data from the UCI Machine Learning Repository. The datasets used include Heart Attack Prediction and Breast Cancer (Wisconsin). 

The seven ML algorithms used are:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Decision Trees
5. Random Forest
6. Gradient Boosting
7. XGBoost

##### Interactive Side-Dashboard
An interactive side-dashboard, created using the `st.sidebar` call in Streamlit, allows users to:
- Choose the dataset (Heart Attack or Breast Cancer).
- Select the ML algorithm (e.g., Logistic Regression, KNN, SVM, etc.).
- Adjust model parameters such as Learning Rate, Random State, Regularization Coefficient, Gamma, Kernel, `n_estimators`, and more.

After selecting the parameters, the app trains the model, and the user can observe real-time changes in:
- **Classification Plot**: Visual representation of the model's predictions.
- **Confusion Matrix**: A summary of prediction results, showing true positives, true negatives, false positives, and false negatives.
- **Model Metrics**: Including Accuracy, Precision, Recall, F1-Score, Mean Squared Error, and Execution Time.

##### Model Performance
The table below shows the accuracy of each model for the Heart Attack and Breast Cancer datasets:

| Model                | Accuracy (Heart Attack) | Accuracy (Breast Cancer) |
|----------------------|-------------------------|--------------------------|
| Logistic Regression  | 91.803%                 | 100.0%                   |
| KNN                  | 86.89%                  | 96.49%                   |
| SVM                  | 93.44%                  | 100.0%                   |
| Decision Trees       | 52.56%                  | 60.53%                   |
| Random Forest        | 90.164%                 | 98.24%                   |
| Gradient Boosting    | 88.53%                  | 96.49%                   |
| XGBoost              | 95.08%                  | 94.737%                  |

This comparison helps users understand the performance of different algorithms and the impact of hyperparameter tuning on model accuracy.

#### 2. User Prediction
In this section, users can use any of the trained models to predict their risk status (High Risk or Low Risk) for heart attacks or breast cancer based on their own input values.

##### User Interaction
Users enter their specific attributes (e.g., age, sex, heart rate, blood sugar) into the app. The chosen model then uses this data to make a prediction about their risk level, providing an immediate and personalized assessment.

### Visual Illustrations
The app includes visual elements such as plots and metrics, allowing users to see real-time updates and understand how different models and parameters affect predictions.


### How We Built It

1. **Technology Stack:** We developed the app using Python and Streamlit, prioritizing an interactive and user-friendly interface for seamless user experience.

2. **Machine Learning Integration:** The backend incorporates seven machine learning algorithms—Logistic Regression, KNN, SVM, Decision Trees, Random Forest, Gradient Boosting, and XGBoost—offering robust predictive capabilities.

3. **Data Utilization:** We trained our models using datasets from the UCI Machine Learning Repository, specifically the Heart Attack Prediction and Breast Cancer (Wisconsin) datasets, to ensure accurate and relevant health predictions.

4. **User Interaction:** The app provides an interactive sidebar where users can select the dataset, choose the desired machine learning algorithm, and fine-tune model parameters, allowing for personalized and dynamic exploration of the app's capabilities.

## Challenges we ran into
One of the significant challenges was ensuring the accuracy and reliability of the predictions across different algorithms. Balancing model complexity with performance and user interpretability was also crucial. Additionally, integrating real-time updates for model metrics and visualizations while maintaining a smooth user experience posed technical hurdles.

## Accomplishments that we're proud of
We are proud of successfully creating an interactive application that combines advanced machine learning techniques with an intuitive user interface. Achieving high accuracy rates for our models, especially 100% accuracy for some algorithms on the Breast Cancer dataset, was a significant milestone. The app's ability to provide real-time feedback and educate users on the impact of different health factors is a testament to our efforts.

## What we learned
Throughout the development process, we gained deeper insights into the intricacies of different machine learning algorithms and their applications in healthcare. We learned the importance of hyperparameter tuning and how it can drastically affect model performance. Additionally, we understood the value of creating user-friendly interfaces that make complex technologies accessible to non-experts.

## What's next for Heart Attack & Breast Cancer Prediction
Future enhancements for the app include expanding the range of health conditions it can predict, incorporating more advanced algorithms, and improving the user interface for better engagement. We aim to integrate personalized health recommendations based on the predictions to help users take actionable steps towards improving their health. Collaborating with healthcare professionals to validate and refine our models further will also be a priority.
