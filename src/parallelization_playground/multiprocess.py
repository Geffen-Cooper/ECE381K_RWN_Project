import multiprocessing
import time


# def test(i, num1, num2, sleep_time):
#     start_time = time.perf_counter()
#     print("start time of process " + str(i) + ": " + str(start_time))
#     time.sleep(sleep_time)
#     product = num1*num2
#     print(product)
#     print("end time of process " + str(i) + ": " + str(time.perf_counter()))

def test(input_array):
    start_time = time.perf_counter()
    print("start time of process " + str(input_array[0]) + ": " + str(start_time))
    time.sleep(input_array[3])
    product = input_array[1]*input_array[2]
    print(product)
    print("end time of process " + str(input_array[0]) + ": " + str(time.perf_counter()))

# def test_2(i, num1, num2):
#     start_time = time.perf_counter()
#     print("start time of process " + str(i) + ": " + str(start_time))
#     time.sleep(1)
#     product = num1*num2
#     print(product)
#     print("end time of process " + str(i) + ": " + str(time.perf_counter()))

'''
def serial_proc():
    start_time = time.perf_counter()
    parallel_0 = multiprocessing.Process(target=test(0, 5, 5))
    parallel_1 = multiprocessing.Process(target=test(1, 6, 6))
    parallel_2 = multiprocessing.Process(target=test(2, 7, 7))
    parallel_3 = multiprocessing.Process(target=test(3, 8, 8))
    parallel_4 = multiprocessing.Process(target=test_2(4, 4, 4))
    parallel_0.start()
    parallel_1.start()
    parallel_2.start()
    parallel_3.start()
    parallel_4.start()
    print("Time in serial: ", time.perf_counter() - start_time)

'''
def parallel_proc():
    start_time = time.perf_counter()
    # parallel_0 = multiprocessing.Process(target=test(0, 5, 5))
    # parallel_1 = multiprocessing.Process(target=test(1, 6, 6))
    # parallel_2 = multiprocessing.Process(target=test(2, 7, 7))
    # parallel_3 = multiprocessing.Process(target=test(3, 8, 8))
    # parallel_4 = multiprocessing.Process(target=test_2(4, 4, 4))
    pool = multiprocessing.Pool(4)
    work = ([0, 5, 5, 2], [1, 6, 6, 2], [2, 7, 7, 2], [3, 8, 8, 2], [4, 4, 4, 1])
    pool.map(test, work)
    pool.close()
    pool.join()
    print("Time in parallel: ", time.perf_counter() - start_time)


def parallel_run(data):
    print("process %s is waiting %s seconds" % (data[0], data[1]))
    time.sleep(int(data[1]))
    print("process " + str(data[0]) + " finished at time: " + str(time.perf_counter()))

# def serial_run(data):
#     print("process %s is waiting %s seconds" % (data[0], data[1]))
#     time.sleep(int(data[1]))
#     print("process " + str(data[0]) + " finished at time: " + str(time.perf_counter()))


print("CPU cores count: " + str(multiprocessing.cpu_count()))
# serial_proc()

parallel_proc()
print("Expected serial time is 9 seconds")
# start_time = time.perf_counter()
# work = (["A",5], ["B", 2], ["C", 1], ["D", 3])
# p = multiprocessing.Pool(4)
# p.map(parallel_run, work)
# print("Parallel Time: " + str(time.perf_counter() - start_time))
# print("Expected serial time is 11 seconds")


'''
serial_time = time.perf_counter()
work = (["A",5], ["B", 2], ["C", 1], ["D", 3])
pool = multiprocessing.Pool(4)
pool.apply_async(serial_run, args=(work))
pool.close()
pool.join()
print("Parallel Time: " + str(time.perf_counter() - serial_time))
'''