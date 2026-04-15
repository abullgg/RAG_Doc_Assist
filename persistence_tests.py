import subprocess
import time
import requests
import os
import shutil

# 1. Clear data
if os.path.exists("data"):
    shutil.rmtree("data")
os.makedirs("data", exist_ok=True)

# Helper to start server
def start_server():
    print("--- Starting Server ---")
    # Use a specific port to avoid conflicts
    return subprocess.Popen(["python", "-m", "uvicorn", "src.main:app", "--port", "8055"])

def wait_for_server():
    for _ in range(20):
        try:
            requests.get("http://127.0.0.1:8055/docs")
            print("Server is up and ready!")
            time.sleep(1) # Extra buffer for startup logic
            return
        except:
            time.sleep(1)
    raise Exception("Server failed to start")

try:
    print("\n=== TEST 2: Starting Fresh Server ===")
    process = start_server()
    wait_for_server()

    print("\n=== TEST 3: Upload First Document ===")
    with open("test.txt", "w") as f:
        f.write("Python is a high-level programming language created in 1991. FastAPI is a modern web framework for building APIs with Python.")
        
    res = requests.post("http://127.0.0.1:8055/upload", files={"file": open("test.txt", "rb")})
    print("Upload 1 Response:", res.json())

    print("\n=== TEST 4: Verifying Persistent Files Created ===")
    print("faiss_index.bin exists:", os.path.exists("data/faiss_index.bin"))
    print("metadata.json exists:", os.path.exists("data/metadata.json"))
    print("chunks.pkl exists:", os.path.exists("data/chunks.pkl"))

    print("\n=== TEST 5: Querying First Document ===")
    res = requests.post("http://127.0.0.1:8055/ask", json={"question": "When was Python created?", "top_k": 3})
    print("Query 1 Response:", res.json())

    print("\n=== TEST 6: Server Restart (Proving Persistence) ===")
    process.kill()
    time.sleep(2)
    process = start_server()
    wait_for_server()

    print("\n=== TEST 7: Query WITHOUT Re-uploading ===")
    res = requests.post("http://127.0.0.1:8055/ask", json={"question": "When was Python created?", "top_k": 3})
    print("Query 2 Response:", res.json())

    print("\n=== TEST 8: Upload Second Document ===")
    with open("test2.txt", "w") as f:
        f.write("Machine learning is a subset of artificial intelligence.")
    res = requests.post("http://127.0.0.1:8055/upload", files={"file": open("test2.txt", "rb")})
    print("Upload 2 Response:", res.json())

    with open("data/metadata.json", "r") as f:
        print("Current Metadata:", f.read())

    print("\n=== TEST 9: Query Newly Uploaded Document ===")
    res = requests.post("http://127.0.0.1:8055/ask", json={"question": "What is machine learning?", "top_k": 3})
    print("Query 3 Response:", res.json())

    print("\n=== TEST 10: Final Restart ===")
    process.kill()
    time.sleep(2)
    process = start_server()
    wait_for_server()

    print("\n=== TEST 11: Final Query on First Doc After Second Restart ===")
    res = requests.post("http://127.0.0.1:8055/ask", json={"question": "When was Python created?", "top_k": 3})
    print("Query 4 Response:", res.json())

finally:
    print("\nCleaning up...")
    process.kill()
    print("FINISHED TESTS ✅")
