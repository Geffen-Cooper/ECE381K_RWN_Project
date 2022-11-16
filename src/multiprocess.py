import multiprocessing
import time


def test():
    start_time = time.perf_counter()
    print("start time of process: " + str(start_time))
    num = 9000 * 10
    num1 = 85 * 10
    print(num)
    print(num1)

def test_2():
    start_time = time.perf_counter()
    print("start time of process: " + str(start_time))
    num = 100 * 100
    num2 = 5*5
    print(num)
    print(num2)



parallel_0 = multiprocessing.Process(target=test())
parallel_1 = multiprocessing.Process(target=test_2())
parallel_2 = multiprocessing.Process(target=test_2())
parallel_3 = multiprocessing.Process(target=test())
parallel_4 = multiprocessing.Process(target=test_2())
parallel_0.start()
parallel_1.start()
parallel_2.start()
parallel_3.start()
parallel_4.start()