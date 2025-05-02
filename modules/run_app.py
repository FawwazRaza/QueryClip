import subprocess
import sys
import os
import time
import threading
import signal
import platform

def run_backend():
    print("Starting FastAPI backend server...")
    if platform.system() == "Windows":
        # Windows version
        return subprocess.Popen(
            ["python", "-m", "uvicorn", "fastapi_backend_updated:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=True
        )
    else:
        # Linux/Mac version
        return subprocess.Popen(
            ["python", "-m", "uvicorn", "fastapi_backend_updated:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

def run_frontend():
    print("Starting Streamlit frontend...")
    if platform.system() == "Windows":
        # Windows version
        return subprocess.Popen(
            ["streamlit", "run", "streamlit_with_error_handling.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=True
        )
    else:
        # Linux/Mac version
        return subprocess.Popen(
            ["streamlit", "run", "streamlit_with_error_handling.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

def print_output(process, name):
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(f"[{name}] {line.strip()}")

def main():
    # Make sure .env file exists
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("GROQ_API_KEY=your_groq_api_key_here\n")
        print("Created a .env file - please edit it to add your GROQ API key before proceeding.")
        return

    # Start backend and frontend
    backend_process = run_backend()
    time.sleep(5)  # Wait for backend to start
    frontend_process = run_frontend()
    
    # Create threads to print output from both processes
    backend_thread = threading.Thread(target=print_output, args=(backend_process, "BACKEND"))
    frontend_thread = threading.Thread(target=print_output, args=(frontend_process, "FRONTEND"))
    
    backend_thread.daemon = True
    frontend_thread.daemon = True
    
    backend_thread.start()
    frontend_thread.start()
    
    print("\n=== RAG Video Chatbot is now running ===")
    print("Backend server: http://localhost:8000")
    print("Frontend app: Check the Streamlit output for the URL (typically http://localhost:8501)")
    print("Press Ctrl+C to stop both servers")
    
    try:
        # Keep the main thread alive
        while backend_process.poll() is None and frontend_process.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        
        # Terminate processes
        if platform.system() == "Windows":
            # Windows requires special handling
            backend_process.terminate()
            frontend_process.terminate()
        else:
            # Unix-like systems
            backend_process.send_signal(signal.SIGTERM)
            frontend_process.send_signal(signal.SIGTERM)
        
        backend_process.wait()
        frontend_process.wait()
        print("Servers stopped.")

if __name__ == "__main__":
    main()