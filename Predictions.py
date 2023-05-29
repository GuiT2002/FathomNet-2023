import tensorflow as tf
from preprocess2 import preprocess
import os
import pandas as pd

model = tf.keras.models.load_model('FN_Model_v1A.h5')

# Load image filenames for prediction
image_filenames = []
for filename in os.listdir('images_eval'):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        image_filenames.append(filename)

# Preprocess the images
images = preprocess('images_eval')  # Replace with your own logic for image preprocessing

# Make predictions
predictions = model.predict(images)

# Format predictions for submission
formatted_predictions = []
for image_filename, prediction in zip(image_filenames, predictions):
    # Get the top 20 predicted categories with highest probability
    top_categories = [str(category) for category in prediction.argsort()[-20:][::-1]]

    # Format the prediction line
    prediction_line = (image_filename, ' '.join(top_categories), 0.5)  # Set 'osd' score to 0.5

    # Add the formatted prediction to the list
    formatted_predictions.append(prediction_line)

# Create DataFrame from formatted predictions
submission_df = pd.DataFrame(formatted_predictions, columns=['id', 'categories', 'osd'])

# Write formatted predictions to a CSV file
submission_df.to_csv('submission2.csv', index=False)
