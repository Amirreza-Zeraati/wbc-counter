# WBC Counter - White Blood Cell Detection & Counting

A Flask-based web application for automated detection and counting of White Blood Cells (WBC) in microscopic blood smear images. This project implements custom image processing algorithms in pure Python, focusing on algorithmic understanding over library dependency.

![WBC Counter Demo](static/demo_preview.png)
*(Note: Add a screenshot of your app here if available)*

## 🚀 Features

- **Automated Counting**: Detects and counts WBCs from raw microscopic images.
- **Adherent Cell Separation**: Uses Advanced Watershed segmentation to accurately separate touching/connected cells.
- **Pure Python Implementation**: Core algorithms like BFS Counting, Median Blur, and Thresholding are implemented from scratch for educational purposes.
- **Web Interface**: Simple and intuitive UI for uploading images and viewing results side-by-side.
- **Batch Processing**: Script included for processing entire datasets automatically.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Image Processing**: OpenCV (for I/O & complex morphology), NumPy (matrix operations)
- **Frontend**: HTML, CSS (Bootstrap)

## 🧩 How It Works (The Pipeline)

The system processes images in three main phases:

### 1. Preprocessing
- **S-Channel Extraction**: Converts image to HSV and uses the Saturation channel to highlight WBCs.
- **Custom Median Blur**: Removes salt-and-pepper noise while preserving edges.
- **Custom Thresholding**: Converts the image to binary (Black & White).
- **Morphological Cleanup**: Uses Opening/Closing to remove small artifacts and fill holes.

### 2. Segmentation (Method 3)
To handle cells that are stuck together:
- **Gradient Subtraction**: Highlights cell centers.
- **Distance Transform**: Calculates the distance of pixels from the background, creating "peaks" at cell centers.
- **Aggressive Thresholding**: Isolates the peaks of the distance map to find distinct markers for each cell.
- **Watershed Algorithm**: Floods the image from these markers to define precise boundaries between touching cells.

### 3. Counting
- **Custom BFS Algorithm**: A manual Breadth-First Search implementation scans the segmented image to identify and count distinct connected components.
- **Area Filtering**: Ignores particles smaller than a set threshold (e.g., 80px) to prevent false positives.

## 📂 Project Structure

```
wbc-counter/
├── app.py                  # Main Flask application
├── batch_process.py        # Script for processing multiple images
├── processing/
│   ├── detector.py         # Main segmentation pipeline (Method 3)
│   └── utilities.py        # Custom pure-Python algo implementations (BFS, Blur, etc.)
├── static/                 # CSS, Uploaded images, Results
├── templates/              # HTML files
└── dataset/                # Folder for input images (for batch processing)
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Amirreza-Zeraati/wbc-counter.git
   cd wbc-counter
   ```

2. Install dependencies:
   ```bash
   pip install flask opencv-python numpy
   ```

### Usage

**Run the Web App:**
1. Start the server:
   ```bash
   python app.py
   ```
2. Open your browser and go to `http://127.0.0.1:5000`
3. Upload an image and see the result!

**Run Batch Processing:**
To process all images in the `dataset/` folder:
```bash
python batch_process.py
```
Results will be saved in `batch_results/`.

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## 📜 License
This project is open-source and available for educational use.
