### This project predicts city-cycle fuel consumption using Deep Learning models based on vehicle attributes.

# Predicting City-Cycle Fuel Consumption Using Vehicle Attributes

Author: Shahd Abu Hassanien 

## Business Problem : 
How can we accurately predict a vehicle’s fuel consumption (in miles per gallon) based on its characteristics?
Efficient fuel consumption prediction helps automotive manufacturers, environmental agencies, and consumers make informed decisions to improve fuel economy, reduce emissions, and lower costs.

## Data:

The dataset is sourced from the University of California, Irvine (UCI) Machine Learning Repository and was introduced in Quinlan (1993).
It contains:

Target variable: City-cycle fuel consumption in miles per gallon (mpg)

## Methods

1. **Data Preprocessing**  
   - Cleaned and preprocessed the dataset to handle missing values and ensure data quality.  
   - Split the dataset into training and testing sets to evaluate model performance effectively.

2. **Model Building**  
   - Created a Sequential model using deep learning techniques suitable for regression tasks.  
   - Incorporated Dropout layers to prevent overfitting and improve generalization.

3. **Training**  
   - Trained the model with Early Stopping to halt training when the validation loss stopped improving, preventing overfitting and saving time.

4. **Evaluation**  
   - Evaluated the trained model using appropriate regression metrics to assess its predictive performance on unseen data.
  


## Results : 

<img width="590" height="1190" alt="image" src="https://github.com/user-attachments/assets/457de8a6-cb54-4b9e-89e8-7a98bdf1d275" />

### Loss Curve During Training:  

This plot shows the training and validation loss across epochs. Both losses decrease consistently, indicating that the model is learning well and generalizing properly to unseen data. The closeness between the training and validation loss curves suggests that the model is not overfitting, and the training process is stable

## Model

The final model is a **Sequential neural network** built using Keras. It consists of the following layers:

- **Input layer**: Matches the number of input features in the dataset.
- **Two hidden layers**: Dense layers with ReLU activation functions to capture complex non-linear relationships in the data.
- **Dropout layer**: Applied after the hidden layers to reduce overfitting by randomly deactivating a fraction of the neurons during training.
- **Output layer**: A single neuron with linear activation to predict fuel consumption (MPG) as a continuous value.

To improve generalization and prevent overfitting, **EarlyStopping** was used to halt training when validation loss stopped improving.

### Performance Metrics

- **Mean Squared Error (MSE)**: 8.59
- **Mean Absolute Error (MAE)**: 2.24
- **R² Score**: _0.892

These metrics demonstrate that the model effectively predicts city-cycle fuel consumption based on a combination of discrete and continuous vehicle attributes.  
The relatively low error values and high R² score suggest strong model performance in solving the business problem of estimating fuel efficiency.


## Recommendations

To further enhance model performance and generalization, the following recommendations are suggested:

- **Feature Engineering**: Create or transform features that may better capture underlying relationships in the data, such as combining related variables or encoding categorical variables differently.
- **Hyperparameter Tuning**: Use techniques like Grid Search or Random Search to optimize the model architecture and training parameters (e.g., number of neurons, learning rate, batch size).
- **Model Comparison**: Experiment with other regression models such as Random Forest, Gradient Boosting, or even more complex neural network architectures to benchmark performance.
- **Cross-Validation**: Incorporate k-fold cross-validation to get a more reliable estimate of model performance.

## Limitations & Next Steps

### Limitations
- The current model was trained on a relatively small dataset, which may limit its ability to generalize to new, unseen data.
- Only a basic neural network was used, without extensive tuning or advanced architectures.
- The model assumes that all input features are relevant and correctly measured, which might not always be the case.

### Next Steps
- **Expand the Dataset**: Incorporate more data samples, if available, to improve learning.
- **Automated Hyperparameter Optimization**: Integrate tools like Optuna or KerasTuner for systematic tuning.
- **Deploy the Model**: Package the model into a deployable API or web app for real-time predictions.
- **Interpretability**: Add model explainability tools like SHAP or LIME to better understand which features influence predictions the most.

## For Further Information

For any additional questions, please contact me:

[![Email](https://img.shields.io/badge/Email-Click%20Here-red?style=for-the-badge&logo=gmail)](mailto:shahdabuhassanien@gmail.com)
