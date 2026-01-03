import requests
import json
import time
import sys

url = "http://localhost:8000/api/chat"
headers = {"Content-Type": "application/json"}
data = {
    "messages": [
        {"role": "user", "content": "Count from 1 to 10 slowly."}
    ]
}

print(f"Sending request to {url}...")
start_time = time.time()
try:
    with requests.post(url, headers=headers, json=data, stream=True) as response:
        print(f"Response status: {response.status_code}")
        response.raise_for_status()
        
        chunk_count = 0
        first_byte_time = None
        
        for line in response.iter_lines():
            if line:
                now = time.time()
                if first_byte_time is None:
                    first_byte_time = now
                    print(f"First byte received after {first_byte_time - start_time:.4f}s")
                
                print(f"Time: {now - start_time:.4f}s | Chunk: {line.decode('utf-8')[:50]}...")
                chunk_count += 1
                
        total_time = time.time() - start_time
        print(f"Total time: {total_time:.4f}s")
        print(f"Total chunks: {chunk_count}")

except Exception as e:
    print(f"Error: {e}")
