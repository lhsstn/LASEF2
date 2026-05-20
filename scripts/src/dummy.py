# dummy_gpu_load.py
import torch
import time
import multiprocessing as mp


def run_dummy(gpu_id: int, matrix_size: int = 24576):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    print(f"[GPU {gpu_id}] Starting dummy workload on {device}")

    # H100이면 float16/bfloat16이 빠르고 부하도 잘 걸림
    dtype = torch.float16

    a = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
    b = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)

    # warm-up
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize(device)

    i = 0
    start = time.time()

    while True:
        c = torch.matmul(a, b)

        # 너무 최적화로 날아가지 않게 가끔 값 사용
        if i % 100 == 0:
            torch.cuda.synchronize(device)
            elapsed = time.time() - start
            # print(f"[GPU {gpu_id}] iter={i}, elapsed={elapsed:.1f}s")

        i += 1


if __name__ == "__main__":
    gpu_ids = [0, 1, 2, 3]

    processes = []
    for gpu_id in gpu_ids:
        p = mp.Process(target=run_dummy, args=(gpu_id,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()