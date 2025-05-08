# Plant-Leaf-Condition-Classifier
This project uses a Vision Transformer (ViT) model to classify plant leaf conditions into various categories such as healthy leaves and diseased leaves. The model is trained on a variety of plant species and their common diseases. After classification, the app provides actionable mitigation strategies to help treat or manage the disease.

Project Notes
Technology Used: Streamlit for the app, TensorFlow for the model, and OpenCV for image processing.
Model Type: Vision Transformer (ViT) trained on a dataset of plant leaf images.
Key Features:
Classify plant leaf conditions.
Suggest mitigation or treatment strategies.
Ensure good image quality (sharpness, lighting, contrast) for accurate predictions.

•	Main Function (main()):

o	Sets the page configuration and title.
o	Loads the model using load_keras_model().
o	Provides instructions for enabling camera access.
o	Handles image input from both file uploads and the camera.
o	Displays the uploaded image.
o	Performs image quality checks.
o	Predicts the class using the loaded model.
o	Displays the prediction, confidence, and mitigation strategy.
o	Saves the image and logs the prediction.
o	Includes a sidebar with project information.
o	Adds project overview to the main body.

I wish to further refine the entire project.

Isaac Baiden

isaac.baiden.stu@uenr.edu.gh

isbaiden.fx@gmail.com

+233 20 8775507

University of Ernergy and Natural Resources, 

Sunyani Ghana

================================================================================
===============================================================================
Dataset  Data for: Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network - Mendeley Data
J, ARUN PANDIAN; GOPAL, GEETHARAMANI (2019), “Data for: Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network”, Mendeley Data, V1, doi: 10.17632/tywbtsjrjv.1
📊 Preprocessing
•	Resizing to 64x64
•	Normalization to [0, 1]
•	Data Augmentation : flipping, rotation
•	Split: 70% training, 15% validation, 15% testing
________________________________________
🧠 2. Model Architecture: Vision Transformer (ViT)
⚙️ Configuration
•	Patch Size: 8
•	Embedding Dimension: 64
•	Attention Heads: 2
•	Transformer Layers: 2
•	Dropout Rate: 0.3
•	Classifier: MLP with two Dense layers
🧩 Custom Layers
A custom PatchExtract layer was defined using tf.image.extract_patches to split images into patches and embed them for attention computation.
🏋️ Training
•	Optimizer: Adam (learning rate = 1e-4)
•	Loss Function: Categorical Crossentropy
•	Epochs: 200
•	Metrics: Accuracy
The model achieved high accuracy on both training and validation sets, outperforming traditional CNN-based baselines.
________________________________________
💻 3. Web Application with Streamlit
A lightweight, interactive Streamlit app was built to:
•	Upload or capture leaf images using the device camera.
•	Run real-time predictions using the trained ViT model.
•	Display:
o	Predicted disease class
o	Confidence score
o	Recommended treatment strategy
•	Store predictions and images in a local log with timestamps.
🧠 Smart Quality Checks:
Before prediction, the app evaluates:
•	Blurriness
•	Lighting
•	Contrast
If poor quality is detected, the user is notified to retake or upload a better image.
________________________________________
🚀 4. Deployment
The app is designed for easy deployment:
•	Lightweight frontend via Streamlit
•	TensorFlow/Keras backend for inference
•	Ready for deployment on:
o	Streamlit Cloud
o	Local LAN/Web server
🔐 Additional Features:
•	Model loading with custom objects
•	CSV Logging of predictions
•	Mitigation strategy database
•	Webcam or file upload support
________________________________________
✅ Outcome
•	Fast, accurate disease detection (real-time).
•	Helps farmers act quickly to prevent spread.
•	Intuitive interface requiring no technical expertise.
•	Easily extendable to new crops or diseases.
________________________________________
🔮 Future Work
•	Integrate with IoT sensors for real-time farm monitoring.
•	Add multi-language support for accessibility.
•	Train a mobile-optimized lightweight model for offline use.


