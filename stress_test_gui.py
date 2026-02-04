import tkinter as tk
from tkinter import ttk
import time
import psutil
import multiprocessing
import numpy as np
import os
import datetime
import csv
import threading

# Global flags
stop_event = threading.Event()
cycle_event = threading.Event()
workers = []
is_running = False

def get_gpu_count():
    """Returns the number of available GPUs (tries OpenGL first, then CUDA)."""
    # Try OpenGL method (works with any GPU)
    try:
        import OpenGL.GL
        import OpenGL.GLUT
        OpenGL.GLUT.glutInit()
        # If GLUT initializes, we have at least one GPU
        return 1  # Assume 1 GPU for OpenGL method
    except:
        pass
    
    # Try PyTorch CUDA method (NVIDIA only)
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    
    return 0

def get_gpu_stats():
    """Returns GPU usage and temperature if available."""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        stats = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            stats.append({'usage': util.gpu, 'temp': temp})
        pynvml.nvmlShutdown()
        return stats
    except:
        return []

def gpu_stress_worker(gpu_id, intensity_percent=100):
    """
    Stable GPU Stress using Standard GLUT Callbacks.
    Switching to glutMainLoop() prevents the window from vanishing.
    """
    # Debug logging
    def log_gpu(msg):
        try:
            with open(f"gpu_debug_{gpu_id}.log", "a") as f:
                f.write(f"{datetime.datetime.now()}: {msg}\n")
        except: pass

    log_gpu("Worker started (Callback Mode)")
    
    try:
        import OpenGL.GL as gl
        import OpenGL.GLUT as glut
        import sys
        
        # State variables
        list_id = 0
        
        def draw():
            # Actual stress rendering
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            
            # Load factor based on intensity
            # If 100%, we draw MORE times.
            loops = int(10 + (intensity_percent / 100.0) * 40) # 10 to 50 loops
            
            for _ in range(loops):
                gl.glCallList(list_id)
                
            glut.glutSwapBuffers()
            # Request next frame immediately for 100% load
            glut.glutPostRedisplay()

        def setup_gl():
            nonlocal list_id
            log_gpu("Setting up GL")
            gl.glClearColor(0.1, 0.1, 0.1, 1.0)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            
            list_id = gl.glGenLists(1)
            gl.glNewList(list_id, gl.GL_COMPILE)
            gl.glBegin(gl.GL_QUADS)
            
            # 50 Quads is SAFE but heavy enough when looped
            for i in range(50):
                # Random-ish colors
                gl.glColor4f((i%10)*0.1, (i%5)*0.2, 0.5, 0.1)
                gl.glVertex2f(-1.0, -1.0)
                gl.glVertex2f(1.0, -1.0)
                gl.glVertex2f(1.0, 1.0)
                gl.glVertex2f(-1.0, 1.0)
            gl.glEnd()
            gl.glEndList()
            log_gpu("Display list Ready")

        # 1. Init Window
        if not hasattr(sys, 'argv') or not sys.argv: sys.argv = ['']
        glut.glutInit(sys.argv)
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB)
        glut.glutInitWindowSize(300, 300)
        glut.glutInitWindowPosition(50, 50)
        glut.glutCreateWindow(b"Stable_GPU_Stress")
        
        # 2. Register Callbacks
        glut.glutDisplayFunc(draw)
        
        # 3. Setup and Run
        setup_gl()
        log_gpu("Entering MainLoop")
        
        # This will run forever until the process is terminated
        glut.glutMainLoop()

    except Exception as e:
        log_gpu(f"CRASH: {e}")
        print(f"GPU Error: {e}")


def cpu_stress_worker(core_index, intensity_percent=100):
    """Worker function to stress a single CPU core with adjustable intensity."""
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS) 
    except:
        pass

    # Calculate work/sleep ratio based on intensity
    work_time = intensity_percent / 100.0
    sleep_time = (100 - intensity_percent) / 100.0 * 0.05  # Max 50ms sleep per cycle
    
    matrix_size = 500
    while not stop_event.is_set():
        start = time.perf_counter()
        while (time.perf_counter() - start) < work_time * 0.1:  # Work for a fraction of time
            A = np.random.rand(matrix_size, matrix_size)
            B = np.random.rand(matrix_size, matrix_size)
            np.dot(A, B)
        if sleep_time > 0:
            time.sleep(sleep_time)

def ram_stress_worker(target_percent=50.0):
    """Worker function to stress RAM by allocating multiple chunks."""
    try:
        total_ram = psutil.virtual_memory().total
        target_bytes = int(total_ram * (target_percent / 100.0))
        
        # Allocate in 256MB chunks for stability
        chunk_size = 256 * 1024 * 1024 
        num_chunks = target_bytes // chunk_size
        
        print(f"RAM Worker: Target {target_bytes / (1024**3):.2f} GB ({target_percent}%)")
        
        memory_chunks = []
        # Total needed chunks
        for i in range(num_chunks):
            if stop_event.is_set(): break
            try:
                # Create and FILL the memory to force actual physical allocation
                chunk = np.zeros(chunk_size // 8, dtype='float64')
                chunk.fill(1.234) # Dirty the pages
                memory_chunks.append(chunk)
                
                # Slow down allocation slightly for system stability
                if i % 2 == 0: 
                    time.sleep(0.05)
            except MemoryError:
                break
                
        print(f"RAM Worker: Allocated {len(memory_chunks) * 256} MB")
        
        while not stop_event.is_set():
            # Randomly touch chunks to keep them in physical RAM
            for chunk in memory_chunks:
                if stop_event.is_set(): break
                chunk[0] = time.time() 
            time.sleep(1.0)
            
    except Exception as e:
        print(f"RAM Worker Error: {e}")


def logger_worker(file_path):
    """Logs system stats to CSV."""
    gpu_mon = False
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_mon = True
        device_count = pynvml.nvmlDeviceGetCount()
    except:
        gpu_mon = False

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Timestamp", "CPU_Usage_Percent", "RAM_Usage_Percent", "RAM_Available_GB"]
        if gpu_mon:
            for i in range(device_count):
                header.extend([f"GPU_{i}_Usage_%", f"GPU_{i}_Temp_C"])
        writer.writerow(header)
        
        while not stop_event.is_set():
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cpu_pct = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            
            row = [timestamp, cpu_pct, ram.percent, round(ram.available/1024**3, 2)]
            
            if gpu_mon:
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    row.extend([util.gpu, temp])
            
            writer.writerow(row)
            f.flush()
    
    if gpu_mon:
        pynvml.nvmlShutdown()


class StressTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PC Cooling Stress Tester")
        self.root.geometry("600x600")
        self.root.resizable(False, False)
        
        # Variables
        self.is_running = False
        self.workers = []
        self.log_thread = None
        self.log_file = None
        self.cycle_thread = None
        
        # Control variables
        self.cpu_intensity = tk.IntVar(value=100)
        self.ram_intensity = tk.IntVar(value=50)
        self.gpu_intensity = tk.IntVar(value=100)
        self.cycle_mode = tk.BooleanVar(value=False)
        self.cycle_on_time = tk.IntVar(value=60)
        self.cycle_off_time = tk.IntVar(value=60)
        
        # Main container
        main_frame = tk.Frame(root, bg="#1e1e1e", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = tk.Label(main_frame, text="System Stress Monitor", 
                        font=("Segoe UI", 20, "bold"), bg="#1e1e1e", fg="#ffffff")
        title.pack(pady=(0, 15))
        
        # Status indicators frame
        stats_frame = tk.Frame(main_frame, bg="#2d2d2d", padx=15, pady=15)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Status
        self.status_label = tk.Label(stats_frame, text="Status: Idle", 
                                    font=("Segoe UI", 16, "bold"), bg="#2d2d2d", fg="#9E9E9E")
        self.status_label.pack(expand=True)
        
        # Settings frame
        settings_frame = tk.Frame(main_frame, bg="#2d2d2d", padx=15, pady=15)
        settings_frame.pack(fill=tk.BOTH, pady=(0, 15))
        
        # CPU Intensity
        cpu_frame = tk.Frame(settings_frame, bg="#2d2d2d")
        cpu_frame.pack(fill=tk.X, pady=5)
        tk.Label(cpu_frame, text="CPU Load:", font=("Segoe UI", 10), 
                bg="#2d2d2d", fg="#ffffff", width=12, anchor="w").pack(side=tk.LEFT)
        self.cpu_slider = tk.Scale(cpu_frame, from_=10, to=100, orient=tk.HORIZONTAL,
                                  variable=self.cpu_intensity, bg="#2d2d2d", fg="#4CAF50",
                                  highlightthickness=0, troughcolor="#1a1a1a", 
                                  activebackground="#4CAF50", sliderrelief=tk.FLAT,
                                  length=250, width=20, showvalue=0)
        self.cpu_slider.pack(side=tk.LEFT, padx=5)
        self.cpu_value_label = tk.Label(cpu_frame, text="100%", font=("Segoe UI", 11, "bold"),
                                       bg="#2d2d2d", fg="#4CAF50", width=6)
        self.cpu_value_label.pack(side=tk.LEFT, padx=5)
        self.cpu_intensity.trace_add('write', lambda *args: self.cpu_value_label.config(text=f"{self.cpu_intensity.get()}%"))
        
        # RAM Intensity
        ram_frame = tk.Frame(settings_frame, bg="#2d2d2d")
        ram_frame.pack(fill=tk.X, pady=5)
        tk.Label(ram_frame, text="RAM Load:", font=("Segoe UI", 10), 
                bg="#2d2d2d", fg="#ffffff", width=12, anchor="w").pack(side=tk.LEFT)
        self.ram_slider = tk.Scale(ram_frame, from_=10, to=90, orient=tk.HORIZONTAL,
                                  variable=self.ram_intensity, bg="#2d2d2d", fg="#2196F3",
                                  highlightthickness=0, troughcolor="#1a1a1a",
                                  activebackground="#2196F3", sliderrelief=tk.FLAT,
                                  length=250, width=20, showvalue=0)
        self.ram_slider.pack(side=tk.LEFT, padx=5)
        self.ram_value_label = tk.Label(ram_frame, text="50%", font=("Segoe UI", 11, "bold"),
                                       bg="#2d2d2d", fg="#2196F3", width=6)
        self.ram_value_label.pack(side=tk.LEFT, padx=5)
        self.ram_intensity.trace_add('write', lambda *args: self.ram_value_label.config(text=f"{self.ram_intensity.get()}%"))
        
        # GPU Intensity
        gpu_frame = tk.Frame(settings_frame, bg="#2d2d2d")
        gpu_frame.pack(fill=tk.X, pady=5)
        tk.Label(gpu_frame, text="GPU Stress:", font=("Segoe UI", 10), 
                bg="#2d2d2d", fg="#ffffff", width=12, anchor="w").pack(side=tk.LEFT)
        self.gpu_slider = tk.Scale(gpu_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                  variable=self.gpu_intensity, bg="#2d2d2d", fg="#FFC107",
                                  highlightthickness=0, troughcolor="#1a1a1a",
                                  activebackground="#FFC107", sliderrelief=tk.FLAT,
                                  length=250, width=15, showvalue=0)
        self.gpu_slider.pack(side=tk.LEFT, padx=5)
        self.gpu_value_label = tk.Label(gpu_frame, text="100%", font=("Segoe UI", 11, "bold"),
                                       bg="#2d2d2d", fg="#FFC107", width=6)
        self.gpu_value_label.pack(side=tk.LEFT, padx=5)
        self.gpu_intensity.trace_add('write', lambda *args: self.gpu_value_label.config(text=f"{self.gpu_intensity.get()}%"))

        
        # Cycle Mode
        cycle_check_frame = tk.Frame(settings_frame, bg="#2d2d2d")
        cycle_check_frame.pack(fill=tk.X, pady=10)
        self.cycle_check = tk.Checkbutton(cycle_check_frame, text="Cycle Mode (ON/OFF intervals)",
                                         variable=self.cycle_mode, bg="#2d2d2d", fg="#ffffff",
                                         selectcolor="#404040", font=("Segoe UI", 10),
                                         activebackground="#2d2d2d", activeforeground="#ffffff",
                                         command=self.toggle_cycle_inputs)
        self.cycle_check.pack(anchor=tk.W)
        
        # Cycle Times
        cycle_times_frame = tk.Frame(settings_frame, bg="#2d2d2d")
        cycle_times_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(cycle_times_frame, text="ON Time (s):", font=("Segoe UI", 9), 
                bg="#2d2d2d", fg="#888888", width=12, anchor="w").pack(side=tk.LEFT)
        self.on_time_entry = tk.Entry(cycle_times_frame, textvariable=self.cycle_on_time,
                                     width=8, font=("Segoe UI", 9), state=tk.DISABLED)
        self.on_time_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Label(cycle_times_frame, text="OFF Time (s):", font=("Segoe UI", 9), 
                bg="#2d2d2d", fg="#888888", width=12, anchor="w").pack(side=tk.LEFT, padx=(15,0))
        self.off_time_entry = tk.Entry(cycle_times_frame, textvariable=self.cycle_off_time,
                                      width=8, font=("Segoe UI", 9), state=tk.DISABLED)
        self.off_time_entry.pack(side=tk.LEFT, padx=5)
        
        # Control button
        self.control_btn = tk.Button(main_frame, text="START STRESS TEST", 
                                    command=self.toggle_stress,
                                    font=("Segoe UI", 14, "bold"),
                                    bg="#4CAF50", fg="white",
                                    activebackground="#45a049",
                                    relief=tk.FLAT,
                                    cursor="hand2",
                                    padx=20, pady=15)
        self.control_btn.pack(fill=tk.X)
        
        # Log file info
        self.log_info = tk.Label(main_frame, text="", 
                                font=("Segoe UI", 9), bg="#1e1e1e", fg="#666666")
        self.log_info.pack(pady=(10, 0))
        
    
    def toggle_cycle_inputs(self):
        """Enable/disable cycle time inputs based on checkbox."""
        if self.cycle_mode.get():
            self.on_time_entry.config(state=tk.NORMAL)
            self.off_time_entry.config(state=tk.NORMAL)
        else:
            self.on_time_entry.config(state=tk.DISABLED)
            self.off_time_entry.config(state=tk.DISABLED)
    
    def toggle_stress(self):
        if not self.is_running:
            self.start_stress()
        else:
            self.stop_stress()
    
    def start_stress(self):
        global stop_event, cycle_event
        
        self.is_running = True
        stop_event.clear()
        cycle_event.clear()
        
        # Disable controls
        self.cpu_slider.config(state=tk.DISABLED)
        self.ram_slider.config(state=tk.DISABLED)
        self.gpu_slider.config(state=tk.DISABLED)
        self.cycle_check.config(state=tk.DISABLED)
        self.on_time_entry.config(state=tk.DISABLED)
        self.off_time_entry.config(state=tk.DISABLED)
        
        # Update button appearance
        self.control_btn.config(text="STOP STRESS TEST", bg="#f44336", activebackground="#da190b")
        self.status_label.config(text="Status: Running", fg="#4CAF50")
        
        # Generate log filename
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"stress_log_{timestamp_str}.csv"
        self.log_info.config(text=f"Logging to: {self.log_file}")
        
        # Start logger
        self.log_thread = threading.Thread(target=logger_worker, args=(self.log_file,))
        self.log_thread.start()
        
        # Start in cycle or continuous mode
        if self.cycle_mode.get():
            self.cycle_thread = threading.Thread(target=self.run_cycle_mode)
            self.cycle_thread.start()
        else:
            self.start_workers()
    
    def run_cycle_mode(self):
        """Run stress test in cycle mode."""
        while not stop_event.is_set():
            # ON CYCLE
            self.status_label.config(text="Status: LOAD ON", fg="#FF5722")
            self.start_workers()
            
            time.sleep(self.cycle_on_time.get())
            
            if stop_event.is_set():
                break
            
            # OFF CYCLE
            self.status_label.config(text="Status: LOAD OFF", fg="#FFC107")
            self.stop_workers()
            
            time.sleep(self.cycle_off_time.get())
    
    def start_workers(self):
        """Start all worker processes."""
        cpu_load = self.cpu_intensity.get()
        ram_load = self.ram_intensity.get()
        gpu_load = self.gpu_intensity.get()
        
        # CPU Workers
        for i in range(psutil.cpu_count()):
            p = multiprocessing.Process(target=cpu_stress_worker, args=(i, cpu_load))
            p.start()
            self.workers.append(p)
        
        # RAM Worker
        ram_p = multiprocessing.Process(target=ram_stress_worker, args=(ram_load,))
        ram_p.start()
        self.workers.append(ram_p)
        
        # GPU Workers
        gpu_count = get_gpu_count()
        if gpu_count > 0:
            for i in range(gpu_count):
                gp = multiprocessing.Process(target=gpu_stress_worker, args=(i, gpu_load))
                gp.start()
                self.workers.append(gp)
    
    def stop_workers(self):
        """Stop all worker processes."""
        for w in self.workers:
            w.terminate()
            w.join()
        self.workers = []
    
    def stop_stress(self):
        global stop_event
        
        self.is_running = False
        self.status_label.config(text="Status: Stopping...", fg="#FFC107")
        
        # Disable button to prevent double-click
        self.control_btn.config(state=tk.DISABLED)
        
        # Run cleanup in background thread to prevent GUI lag
        cleanup_thread = threading.Thread(target=self._cleanup_workers)
        cleanup_thread.start()
    
    def _cleanup_workers(self):
        """Background cleanup of worker processes."""
        global stop_event
        
        # Stop all workers
        stop_event.set()
        
        self.stop_workers()
        
        if self.log_thread:
            self.log_thread.join()
        
        if self.cycle_thread:
            self.cycle_thread.join()
        
        # Update GUI in main thread
        self.root.after(0, self._finalize_stop)
    
    def _finalize_stop(self):
        """Finalize stop operation (runs in main GUI thread)."""
        # Re-enable controls
        self.cpu_slider.config(state=tk.NORMAL)
        self.ram_slider.config(state=tk.NORMAL)
        self.gpu_slider.config(state=tk.NORMAL)
        self.cycle_check.config(state=tk.NORMAL)
        self.toggle_cycle_inputs()
        
        # Update button appearance
        self.control_btn.config(text="START STRESS TEST", bg="#4CAF50", 
                               activebackground="#45a049", state=tk.NORMAL)
        self.status_label.config(text="Status: Idle", fg="#9E9E9E")
        self.log_info.config(text=f"Test complete. Data saved to: {self.log_file}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    root = tk.Tk()
    app = StressTestGUI(root)
    root.mainloop()
