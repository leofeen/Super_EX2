import matplotlib.pyplot as plt

from generate_jobs import JOB_CONFIGS

NUMBER_OF_BATCHES = 5
COMMON_BATCH_PATH = r"./batch_"


def compute_and_write_omp_averages() -> None:
    omp_jobs = JOB_CONFIGS['OMP']
    with open("averages.txt", "w") as file:
        file.write("M, N, threads, average, iterations, max-min, acceleration\n")

    for key, threads in omp_jobs.items():
        m, n = key.split()
        grid_max_time = 0
        for thread in threads:
            time_sum = 0
            iterations_sum = 0
            min_time = 100000000
            max_time = -1
            for i in range(1, NUMBER_OF_BATCHES+1):
                with open(COMMON_BATCH_PATH + str(i) + rf"/result_{m}_{n}_{thread}.txt", "r") as file:
                    time, iterations = map(float, file.readline().split())
                    time_sum += time
                    iterations_sum += iterations

                    if time < min_time:
                        min_time = time

                    if time > max_time:
                        max_time = time

            with open("averages.txt", "a") as file:
                average_time = time_sum/NUMBER_OF_BATCHES
                if grid_max_time == 0:
                    grid_max_time = average_time
                file.write(f"{m} {n} {thread} {average_time:5f} {iterations_sum/NUMBER_OF_BATCHES} {max_time-min_time} {grid_max_time/average_time:5f}\n")


def plot_acceleration() -> None:
    omp_jobs = JOB_CONFIGS['OMP']
    
    with open("./averages.txt", "r") as file:
        file.readline()
        for key, threads in omp_jobs.items():
            m, n = key.split()
            mn_result = []
            for _ in threads:
                _, _, _, _, _, _, acc = map(float, file.readline().split())
                mn_result.append(acc)

            plt.plot(threads, mn_result, label=f"{m}x{n}")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # compute_and_write_omp_averages()
    plot_acceleration()