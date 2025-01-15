# Weather_Image_Recognition
Analyzing weather images to determine weather condition using Machine Learning.

---

# Weather Condition Classification using VGG16

This project implements a weather condition classification model using the VGG16 architecture, customized with additional layers for fine-tuning. The model is trained to classify images into 11 weather categories, leveraging transfer learning, data augmentation, and modern deep learning techniques.

## Features
- **Pre-trained VGG16 Backbone:** The VGG16 model (trained on ImageNet) is used as the base model for feature extraction.
- **Custom Classification Layers:** Dense, Batch Normalization, and Dropout layers are added for optimal classification performance.
- **Data Augmentation:** Images are augmented with transformations like rotation, width/height shifts, and horizontal flips.
- **Learning Rate Scheduler:** Dynamically adjusts the learning rate during training for improved convergence.
- **Model Checkpointing:** Automatically saves the best model based on validation accuracy.
- **Supports Class Imbalance Handling:** Includes class distribution analysis for better insights.

## Project Structure
- **Dataset Splitting:** The dataset is split into training, validation, and test sets with a 5%-95% split ratio for flexibility.
- **Image Preprocessing:** Images are resized to 224x224 pixels, normalized, and converted to RGB format.
- **Class Distribution Analysis:** Includes tools to visualize and analyze the class distribution across the dataset.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/your-username/weather-classification.git
   cd weather-classification
   ```
2. Install required libraries:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. **Dataset Preparation:**
   - Organize your dataset into subfolders for each class (e.g., `dew`, `fogsmog`, `frost`).
   - Place the dataset in the specified directory structure or adjust the paths in the code.

2. **Run the Preprocessing Script:**
   - Preprocess the dataset and split it into training, validation, and test sets.

3. **Train the Model:**
   - Train the VGG16-based model using the script provided:
     ```
     python train.py
     ```
   - Monitor training accuracy and validation performance.

4. **Evaluate the Model:**
   - Use the saved model to evaluate performance on the test set or for inference.

## Results
- **Training Accuracy:** Achieved after 10 epochs with data augmentation and learning rate scheduling.
- **Validation Accuracy:** Automatically monitored with model checkpointing to save the best-performing model.

## Dependencies
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn
- OpenCV
- PIL (Pillow)

## Acknowledgments
- VGG16 model and weights are based on the TensorFlow/Keras implementation.
- The dataset and weather categories are inspired by common meteorological conditions.

## Contributions
Feel free to fork this repository, submit issues, or suggest improvements.

## License
This project is open-source and available under the [MIT License](LICENSE).

---

This description is ready to be pasted directly into a README file on GitHub. Adjust the repository URL and license information as needed.
