# System Stress Test Tool

A robust, Python-based system stress testing utility designed to generate significant thermal and computational load on CPU, RAM, and GPU components. It features a dark-mode GUI, automated duty cycles, and comprehensive logging.

## 🚀 Key Features

*   **Universal GPU Stress**:
    *   Uses **OpenGL Display Lists** and massive overdraw to stress GPU fill-rate and memory bandwidth.
    *   **Hybrid Load Balancing**: Automatically adjusts load (frame loops) based on intensity settings.
    *   **Stability**: Uses standard GLUT callbacks (`glutMainLoop`) to prevent window freezing or driver crashes (TDR), making it safe for both weak integrated graphics (iGPU) and high-end dedicated cards (dGPU like RTX series).
    *   **No Vendor Lock-in**: Works on NVIDIA, AMD, and Intel graphics without requiring CUDA.
*   **Multi-Core CPU Stress**:
    *   Utilizes Python's `multiprocessing` to bypass the GIL (Global Interpreter Lock) and saturated all available CPU cores.
*   **Physical RAM Stress**:
    *   Allocates continuous chunks of memory and "dirties" pages to force physical RAM mapping.
*   **Cycle Mode (Thermal Cycling)**:
    *   Automated **ON/OFF** intervals (e.g., 60s load, 60s idle).
    *   Ideal for testing cooling system recovery and thermal capacity.
*   **Data Logging**:
    *   Records system stats (CPU %, RAM %, GPU Usage/Temp) to a CSV file every second.

## 📋 Requirements

*   **OS**: Windows (Recommended), Linux, or macOS.
*   **Python**: Version 3.8 or higher.

## 📦 Installation & Usage

Since this repository provides the source code, you can run the tool directly with Python.

1.  **Clone or Download** this repository.

2.  **Install Dependencies**:
    Open your terminal or command prompt and install the required libraries:
    ```bash
    pip install psutil numpy PyOpenGL pynvml
    ```
    *   `psutil`: For CPU/RAM monitoring.
    *   `numpy`: For matrix operations (CPU stress).
    *   `PyOpenGL`: For GPU stress rendering.
    *   `pynvml`: (Optional) For NVIDIA GPU monitoring.

3.  **Run the Tool**:
    ```bash
    python stress_test_gui.py
    ```

## 🛠️ Interface Guide

1.  **Sliders**: Adjust the intensity (10-100%) for CPU, RAM, and GPU.
    *   *GPU Note*: At 100%, the tool forces maximum frame redraws with zero sleep time.
2.  **Cycle Mode**: Check the box to enable automated on/off switching.
    *   **ON ms**: Duration of the stress phase (in seconds).
    *   **OFF ms**: Duration of the cooling/idle phase (in seconds).
3.  **Start/Stop**: Click **"START STRESS TEST"** to begin.
    *   A separate GPU window (`Stable_GPU_Stress`) will appear. **Do not close this window manually**; it will close automatically when you stop the test.

## 🏗️ Optional: Building an Executable

If you wish to create a standalone `.exe` file for portability (so you don't need Python installed on the target machine), you can build it yourself using PyInstaller.

1.  **Install PyInstaller**:
    ```bash
    pip install pyinstaller
    ```

2.  **Run the Build Command**:
    ```powershell
    py -m PyInstaller --onefile --windowed --name "StressTestTool" ^
    --hidden-import=psutil --hidden-import=numpy --hidden-import=numpy.core ^
    --hidden-import=OpenGL --hidden-import=OpenGL.GL --hidden-import=OpenGL.GLUT ^
    --hidden-import=OpenGL.GLU --hidden-import=pynvml --clean stress_test_gui.py
    ```

3.  The compiled executable will appear in the `dist/` folder.

## ⚠️ Disclaimer

**Use at your own risk.** This software is designed to maximize power consumption and heat generation.
*   Ensure your cooling system is functional before running at 100% intensity.
*   The author is not responsible for any hardware damage caused by thermal throttling failure or power supply overloading.
