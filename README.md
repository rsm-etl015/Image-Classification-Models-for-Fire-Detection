---

# Fire Detection Model Repository

## Why the Files Are Organized Separately

Due to computational limitations, I was unable to train models on my laptop. Initially, I attempted to use Google Colab, but its runtime limit was too short for my needs. Consequently, I decided to use **Kaggle Notebooks**, which offer longer runtime sessions. However, I exhausted my **30-hour GPU quota early in the week**, forcing me to train models on **CPU only**. To manage this limitation, I had to **reduce the number of epochs** during training.  

This repository is structured to separate training, evaluation, and inference files, ensuring clarity and ease of use.

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
