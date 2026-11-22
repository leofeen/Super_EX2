import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import dataclasses


def check_inside_region(x: float, y: float) -> int:
    if (np.abs(x) >= 2) or (np.abs(y) >= 2) or (x*x + (y-2)*(y-2) < 1):
        return -1
    if np.abs(x*x + (y-2)*(y-2) - 1) < 1e-6:
        return 0
    return 1

@dataclasses.dataclass
class SolutionData:
    M : int
    N : int
    x_min : float
    x_max : float
    y_min : float 
    y_max : float
    data : np.ndarray

def read_data(path: str) -> SolutionData:
    with open(path) as data_file:
        data_file.readline()
        m, n, x_min, x_max, y_min, y_max = map(float, data_file.readline().split())
        M, N = int(m), int(n)
        data = np.empty((M+1, N+1),  dtype=float)
        i = 0
        while line := data_file.readline():
            data[i, :] = list(map(float, line.split()))
            i += 1

    return SolutionData(M, N, x_min, x_max, y_min, y_max, data)

def get_masked_solution_data(solution: SolutionData) -> np.ndarray:
    h_x = (solution.x_max - solution.x_min) / solution.M
    h_y = (solution.y_max - solution.y_min) / solution.N

    data_filter = np.array([[ bool(check_inside_region(solution.x_min + i*h_x, solution.y_min + j*h_y) >= 0) for j in range(solution.N+1)] for i in range(solution.M+1)])
    return np.where(data_filter, solution.data, 0)

def plot_solution(solution: SolutionData) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1 : Axes
    ax2 : Axes

    image = ax1.imshow(solution.data.T, origin='lower', cmap="RdYlBu", extent=(solution.x_min, solution.x_max, solution.y_min, solution.y_max))
    ax1.set_aspect('equal')
    ax1.set_title("Unmasked")

    ax2.imshow(get_masked_solution_data(solution).T, origin='lower', cmap="RdYlBu", extent=(solution.x_min, solution.x_max, solution.y_min, solution.y_max))
    ax2.set_aspect('equal')
    ax2.set_title("Masked")
    cax = fig.add_axes((0.125, 0.1, 0.8-0.025, 0.05))
    fig.colorbar(image, cax=cax, orientation="horizontal")

    plt.show()


def plot_region() -> None:
    M = 500
    x_max = 2
    x_min = -2
    h_x = (x_max - x_min) / M

    data = np.array([[ int(check_inside_region(x_min + i*h_x, x_min + j*h_x) >= 0) for j in range(M+1)] for i in range(M+1)])

    fig, ax = plt.subplots(1, 1)
    image = ax.imshow(data.T, origin="lower", cmap="RdYlBu", extent=(x_min, x_max, x_min, x_max))
    ax.set_aspect('equal')
    # cax = fig.add_axes((0.125, 0.1, 0.8-0.025, 0.05))
    # fig.colorbar(image, orientation="horizontal")

    plt.show()


def read_errors(path: str) -> np.ndarray:
    errs = []
    with open(path, "r") as file:
        while line := file.readline():
            errs.append(float(line))

    return np.array(errs)

def plot_errors(errs: np.ndarray) -> None:
    xs = range(1, len(errs)+1)
    precision = 1e-8
    plt.plot(xs, errs)
    plt.plot(xs, [precision]*len(xs))
    plt.show()


def read_diffs(path: str, abs_: bool = False) -> list[SolutionData]:
    result: list[SolutionData] = []
    with open(path) as data_file:
        m, n, x_min, x_max, y_min, y_max = map(float, data_file.readline().split())
        M, N = int(m), int(n)
        data = np.empty((M+1, N+1),  dtype=float)
        i = 0
        while line := data_file.readline():
            if line == "\n":
                result.append(SolutionData(M, N, x_min, x_max, y_min, y_max, data))
                data = np.empty((M+1, N+1),  dtype=float)
                i = 0
                continue
            if abs_:
                data[i, :] = list(map(lambda x: abs(float(x)), line.split()))
            else:
                data[i, :] = list(map(float, line.split()))
            i += 1

    return result

def plot_diffs(diffs: list[SolutionData]) -> None:
    for diff in diffs:
        plot_solution(diff)


if __name__ == "__main__":
    what_to_draw = input("What do you want to draw? (sol | reg | errs | diffs | diffs_abs) ")
    match what_to_draw:
        case "sol":
            M, N, index = map(int, input("Input M, N and index: ").split())
            file_path = rf"./build/result_{M}_{N}_{index}.txt" 
            plot_solution(read_data(file_path))

        case "reg":
            plot_region()

        case "errs":
            file_path = r"./errs_and_diffs/errors.txt"
            plot_errors(read_errors(file_path))

        case "diffs":
            file_path = r"./errs_and_diffs/diffs.txt"
            plot_diffs(read_diffs(file_path))

        case "diffs_abs":
            file_path = r"./errs_and_diffs/diffs.txt"
            plot_diffs(read_diffs(file_path, abs_=True))

        case _:
            print("Invalid input")