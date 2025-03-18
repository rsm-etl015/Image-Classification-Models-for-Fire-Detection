---

# Fire Detection Model Repository

## Why the Files Are Organized Separately

Due to computational limitations, I was unable to train models on my laptop. Initially, I attempted to use Google Colab, but its runtime limit was too short for my needs. Consequently, I decided to use **Kaggle Notebooks**, which offer longer runtime sessions. However, I exhausted my **30-hour GPU quota early in the week**, forcing me to train models on **CPU only**. To manage this limitation, I had to **reduce the number of epochs** during training.  

This repository is structured to separate training, evaluation, and inference files, ensuring clarity and ease of use.

---
## Dataset: **The Wildfire Dataset**

This project uses the **Wildfire Dataset**, a diverse and evolving dataset designed to improve deep learning-based forest fire detection. The dataset consists of **2,700 aerial and ground-based images**, sourced from public domain platforms like government databases, Flickr, and Unsplash. It aims to represent various **environmental scenarios, forest types, geographical locations, and fire conditions**, making it a strong benchmark for machine learning research in wildfire detection.

### **Dataset Details**
- **Size**: **10.74 GB** (too large to be included in this repository)
- **Source**: Kaggle ([Dataset Link](https://doi.org/10.3390/f14091697))
- **Research Paper**:  
  El-Madafri I, Peña M, Olmedo-Torre N. *The Wildfire Dataset: Enhancing Deep Learning-Based Forest Fire Detection with a Diverse Evolving Open-Source Dataset Focused on Data Representativeness and a Novel Multi-Task Learning Approach.* **Forests. 2023; 14(9):1697**  
  [DOI: 10.3390/f14091697](https://doi.org/10.3390/f14091697)

Since the dataset is large (**10.74 GB**), **it is not included in this repository**.  
To use it, **please download it from Kaggle** and upload it to your Kaggle Notebook.

### **Running the Code**
I strongly recommend running this project on **Kaggle Notebooks** instead of a local machine. The `.ipynb` files require significant computational power, and without a strong GPU, training can be very slow.

---

## Repository Structure

```
📂 Python Notebooks
│   ├── 📂 Training
│   │   ├── <Model_Training_Notebooks>.ipynb  # Jupyter notebooks for training models
│   ├── Evaluation.ipynb  # Notebook to evaluate trained models and generate performance metrics
│   ├── MP4 Video Fire Detection.ipynb  # Notebook for detecting fire in MP4 video files
│   ├── Live Fire Detection.ipynb  # Notebook for real-time fire detection using a webcam
│
📂 Output
│   ├── 📂 <Model_Name>  # Each trained model has its own folder (e.g., BaselineCNN, ResNet50, etc.)
│   │   ├── all_result.json  # Accuracy results for 10 different hyperparameter combinations
│   │   ├── all_history.json  # Loss and accuracy history for the best-performing model
│   │   ├── best_model_info.json  # Hyperparameters of the best model
│   │   ├── best_model.keras  # Saved best model in Keras format
│
├── Input.mp4  # The input video file used for MP4-based fire detection
├── Output Video.mp4  # The output video with detected fire regions
├── DenseNet121.onnx  # The pre-trained model used for both live and video fire detection
```

---

## Description of Key Files

### **Python Notebooks**
- **Training Folder**: Contains all `.ipynb` notebooks for training models with different architectures and hyperparameters.
- **Evaluation.ipynb**: Runs performance evaluation on trained models, including accuracy, loss, and confusion matrices.
- **MP4 Video Fire Detection.ipynb**: Accepts an MP4 video as input and detects fire occurrences in the frames.
- **Live Fire Detection.ipynb**: Uses a webcam to perform real-time fire detection.

### **Output Folder**
Each trained model (e.g., **BaselineCNN, ResNet50**) has a separate folder containing:
- **`all_result.json`**: Stores accuracy metrics for different hyperparameter combinations.
- **`all_history.json`**: Logs loss and accuracy per epoch for the best-performing model.
- **`best_model_info.json`**: Stores hyperparameter details of the best model.
- **`best_model.keras`**: The saved best model for further use.

### **Standalone Files**
- **`Input.mp4`**: The raw input video used for MP4-based fire detection.
- **`Output Video.mp4`**: The processed video with detected fire regions.
- **`DenseNet121.onnx`**: The pre-trained deep learning model used for both live webcam and MP4 fire detection.

---
