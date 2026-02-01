import requests
import time

url = "http://127.0.0.1:8000/chat"
data = {"prompt": "Count to 10 quickly."}

print("Sending request...")
start_time = time.time()
response = requests.post(url, data=data, stream=True)

print(f"Response status: {response.status_code}")

if response.status_code == 200:
    print("Streaming started:")
    first_chunk_time = None
    last_chunk_time = time.time()
    chunk_count = 0
    
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            current_time = time.time()
            if first_chunk_time is None:
                first_chunk_time = current_time
                print(f"Time to first chunk: {first_chunk_time - start_time:.4f}s")
            
            # Print latency between chunks if it's noticeable (emulating generation time)
            diff = current_time - last_chunk_time
            print(f"Chunk received: {chunk.decode('utf-8', errors='ignore')} (latency: {diff:.4f}s)")
            
            last_chunk_time = current_time
            chunk_count += 1
            if chunk_count > 5: # Just check first few chunks
                print("...")
                break
    
    print(f"\nTotal setup time: {first_chunk_time - start_time:.4f}s" if first_chunk_time else "No chunks received")
else:
    print("Error:", response.text)
