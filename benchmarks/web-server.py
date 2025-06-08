from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import threading
import json

class WorkSimulatorHandler(BaseHTTPRequestHandler):
    request_count = 0
    start_time = time.time()

    @classmethod
    def get_rps(cls):
        now = time.time()
        elapsed = now - cls.start_time
        rps = cls.request_count / elapsed if elapsed > 0 else 0
        return {
            "requests": cls.request_count,
            "elapsed_seconds": elapsed,
            "requests_per_second": rps
        }

    def do_GET(self):
        if self.path == '/ping':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'pong')
        elif self.path == '/work':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            start_time = time.time()
            total = 0
            for i in range(100_000_000):  
                total += i
            processing_time = time.time() - start_time
            response = {
                "status": "done",
                "time": time.time(),
                "processing_time": processing_time,
                "total": total  
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
        elif self.path == '/get_rps':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            rps_data = self.get_rps()
            self.wfile.write(json.dumps(rps_data).encode('utf-8'))
        else:
            self.send404()
        self.__class__.request_count += 1


def reward_provider(server):
    with open('/tmp/pipe1', 'w') as f:
        while True:
            rps = WorkSimulatorHandler.get_rps() 
            reward = rps['requests_per_second'] / 1000000
            f.write(str(reward)+'\n')
            f.flush()
            time.sleep(0.1)


def main():
    host = 'localhost'
    port = 8000
    server = HTTPServer((host, port), WorkSimulatorHandler)

    # Start reward provider thread
    reward_sender = threading.Thread(target=reward_provider, args=(server,))
    reward_sender.start()

    print(f'Server started at http://{host}:{port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nServer stopped')
        server.server_close()

if __name__ == '__main__':
    main()
