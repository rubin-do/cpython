import requests
import time
import concurrent.futures

def make_request(url, request_id):
    try:
        start_time = time.time()
        response = requests.get(url)
        end_time = time.time()
        return {
            "id": request_id,
            "status": response.status_code,
            "elapsed": end_time - start_time,
            "url": url
        }
    except Exception as e:
        return {
            "id": request_id,
            "error": str(e),
            "url": url
        }

def load_test(url, num_requests, num_workers):
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_requests):
            futures.append(executor.submit(make_request, url, i))
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    end_time = time.time()
    total_elapsed = end_time - start_time
    rps = num_requests / total_elapsed if total_elapsed > 0 else 0

    print(f"Total requests: {num_requests}")
    print(f"Total elapsed time: {total_elapsed:.2f} seconds")
    print(f"Requests per second: {rps:.2f}")

if __name__ == "__main__":
    endpoint = "http://localhost:8000/work"
    load_test(endpoint, num_requests=2000, num_workers=10)
