import threading
import time
import random
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = threading.Lock()
        self.request_count = 0
        self.start_time = time.time()
        self.last_rps_time = self.start_time
        self.rps = 0

    def get(self, key):
        with self.lock:
            self.request_count += 1
            if key not in self.cache:
                return -1
            value = self.cache.pop(key)
            self.cache[key] = value
            return value

    def put(self, key, value):
        with self.lock:
            self.request_count += 1
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = value

    def get_rps(self):
        with self.lock:
            elapsed = time.time() - self.last_rps_time
            if elapsed == 0:
                return self.rps
            self.rps = self.request_count / elapsed
            self.request_count = 0
            self.last_rps_time = time.time()
            return self.rps


def load_test(cache, key_range):
    while True:
        op = random.choice(['get', 'put'])
        key = random.randint(1, key_range)
        if op == 'get':
            cache.get(key)
        else:
            value = random.randint(1, 1000000)
            cache.put(key, value)
        # time.sleep(0.001)

def reward_provider(cache):
    with open('/tmp/pipe1', 'w') as f:
        while True:
            rps = cache.get_rps() 
            reward = rps / 1000000
            f.write(str(reward)+'\n')
            f.flush()
            time.sleep(0.1)

if __name__ == "__main__":
    cache_capacity = 1000
    key_range = 5000

    cache = LRUCache(cache_capacity)

    # Start reward provider thread
    reward_provider = threading.Thread(target=reward_provider, args=(cache,))
    reward_provider.start()

    load_test(cache, key_range)
