import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Read the dataset
data = pd.read_csv('E:\Desktop/Dataset of Diabetes .csv')

# Drop irrelevant columns like ID and No_Pation
data = data.drop(columns=['ID', 'No_Pation'])

# Encode categorical variables (Gender) to numerical
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Split features and target variable
X = data.drop(columns=['CLASS'])
y = data['CLASS']

# Convert target variable to binary (0 for Non-Diabetic, 1 for Diabetic)
y = (y == 'D').astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Save the trained model
model.save('diabetes_prediction_model.h5')