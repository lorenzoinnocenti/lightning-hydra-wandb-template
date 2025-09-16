import threading
import time
import torch

def start_memory_daemon(sleep_time=10):
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    thread = threading.Thread(target=print_gpu_memory, args=(sleep_time,))
    thread.daemon = True
    thread.start()
    return thread


def print_gpu_memory(sleep_time):
    while True:
        summary = torch.cuda.memory_summary(device="cuda", abbreviated=False)
        # print(summary)
        lines = summary.split('\n')
        for line in lines:
            if 'GPU reserved memory' in line:
                parts = line.split('|')
                cur_usage = parts[2].strip()
                peak_usage = parts[3].strip()
                print("")
                print(f"Current GPU memory usage: {cur_usage}, peak: {peak_usage}")
                break
        time.sleep(sleep_time)
        
        