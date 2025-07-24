# DeepSpermMotility

![Sperm Motility Analysis](images/movilidad.png)

## Introduction
DeepSpermMotility is a project that analyzes sperm motility from videos using deep learning and computer vision techniques. It automates the process of tracking sperm and classifying their movement patterns.

## ✨ Features
*   **Sperm detection and tracking:** Utilizes YOLOv5 for detecting sperm and SORT (Simple Online and Realtime Tracking) for tracking them across video frames.
*   **Calculation of various motility parameters:** Computes key sperm motility metrics such as:
    *   VCL (Curvilinear Velocity)
    *   VSL (Straight-Line Velocity)
    *   VAP (Average Path Velocity)
    *   LIN (Linearity: VSL/VCL)
    *   STR (Straightness: VSL/VAP)
    *   WOB (Wobble: VAP/VCL)
    *   ALH (Amplitude of Lateral Head Displacement)
    *   BCF (Beat Cross Frequency)
*   **Classification of sperm motility:** Classifies sperm into categories (e.g., progressive, hyperactive, immotile) using a pre-trained XGBoost model.
*   **GUI for easy video processing and visualization:** Provides a user-friendly graphical interface to load videos, initiate processing, and view results.
*   **Outputs detailed tracking and feature data:** Generates CSV files containing comprehensive tracking information and calculated motility parameters for each sperm.

## ⚙️ Workflow
1.  **Video Input:** The user selects a video file through the GUI or specifies the path in the command-line script.
2.  **Sperm Detection:** Each frame of the video is processed by the YOLOv5 model to detect the location of sperm heads.
3.  **Sperm Tracking:** The SORT algorithm is applied to the detected sperm across frames to maintain unique identities for each sperm and record their trajectories.
4.  **Parameter Calculation:** Based on the recorded trajectories, various motility parameters (VCL, VSL, VAP, LIN, etc.) are calculated for each tracked sperm.
5.  **Motility Classification:** The calculated motility parameters are fed into a pre-trained XGBoost model, which classifies each sperm's motility type.
6.  **Output Generation:** The system saves the tracking data, calculated features, and velocity information into CSV files. Processed videos showing tracking may also be generated.

## Folder Structure
*   `src/`: Contains all the Python source code for the application, including the GUI, detection, tracking, feature calculation, and classification logic.
*   `models/`: Stores pre-trained machine learning models. This includes the XGBoost classifier for motility.
*   `results/`: Default output directory where CSV files with tracking data, motility features, and processed video data are saved.
*   `data/`: (Implicit) Expected location for input video data. Scripts might reference paths like `../data/VISEM_Tracking/`. Users should place their raw video files here.
*   `YOLO_model/`: (Implicit) Expected location for the YOLOv5 model weights file.

## 🛠️ Setup and Installation
1.  **Python:** Python 3.x is recommended.
2.  **Dependencies:** Install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
3.  **YOLOv5 Model:**
    *   Download the YOLOv5 model weights file, specifically `best_yolov5x.pt`. This model is often used for object detection and might have been fine-tuned for sperm detection in this project.
    *   You can typically find YOLOv5 models on the Ultralytics GitHub repository releases page (e.g., `https://github.com/ultralytics/yolov5/releases`). You may need to search for the specific `best_yolov5x.pt` or a suitable generic version like `yolov5x.pt`.
    *   Create a directory named `YOLO_model/` in the root of the project.
    *   Place the downloaded `.pt` file into the `YOLO_model/` directory.
4.  **Pre-trained Classifier:** The pre-trained XGBoost model for motility classification is included in the `models/` directory.

## ▶️ How to Run

### Using the GUI
1.  Run the application:
    ```bash
    python src/app.py
    ```
2.  In the GUI, click **Select Video** to choose a video file.
3.  Enter a **Test Name** for the output folder.
4.  Click **Start Process** to begin the analysis.
5.  Output files will be saved in a new folder inside the `results` directory, named with the test name you provided.

### Using the Command Line Interface (CLI)
For more advanced users or batch processing, you can run the main processing script directly from the command line.

1.  Ensure all dependencies and models are set up correctly.
2.  Run the script with the following command, providing the video path and a test name:
    ```bash
    python src/sperm_video_classify.py "path/to/your/video.mp4" "your_test_name"
    ```
    - Replace `"path/to/your/video.mp4"` with the actual path to your video file.
    - Replace `"your_test_name"` with a name for your analysis, which will be used for the output files.

## 📄 Output
The application generates the following output files, typically saved in a subdirectory within `results/` named after the `name_video` or test name provided:
*   `tracking_sperm.csv`: Contains the raw tracking data for each sperm, including frame number, sperm ID, and bounding box coordinates.
*   `features_sperm.csv`: Lists the calculated motility parameters for each tracked sperm.
*   `velocity_sperm.csv`: Contains detailed velocity information for each sperm.
*   Processed videos (optional): The system might also output videos with tracking visualizations.

## Dependencies
All required Python packages are listed in the `requirements.txt` file.

## Contributing
Contributions are welcome. Please open an issue to discuss your ideas or submit a pull request with your changes.

## License
This project is currently unlicensed.