import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def read_csv_data(filename):
    data = pd.read_csv(filename)
    return data['Dystans (m)'].to_list(), data['Wysokość (m)'].to_list()


def read_txt_data(filename):
    data = pd.read_csv(filename, sep=' ')
    return data['Dystans_(m)'].to_list(), data['Wysokość_(m)'].to_list()


def fit_data(x_vec, y_vec, x_nodes, y_nodes):
    if x_nodes[-1] != x_vec[-1]:
        x_nodes.append(x_vec[-1])
        y_nodes.append(y_vec[-1])
    return len(x_nodes)


def interpolation(x_vec, y_vec, data_x, n_nodes):
    interpolated_values = []

    for i in range(len(data_x)):
        interpolated_values.append(lagrange_interpolation(x_vec, y_vec, n_nodes, data_x[i]))
    return interpolated_values


def lagrange_interpolation(x_vec, y_vec, n_nodes, x):
    result = 0.0

    for i in range(n_nodes):
        nominator = 1.0
        denominator = 1.0
        for j in range(n_nodes):
            if i != j:
                nominator *= (x - x_vec[j])
                denominator *= (x_vec[i] - x_vec[j])
        result += y_vec[i] * (nominator/denominator)
    return result


def spline_parameters(x_vec, y_vec):
    n_nodes = len(x_vec)
    A = np.zeros((4 * (n_nodes-1), 4 * (n_nodes-1)))
    b = np.zeros(4 * (n_nodes-1))

    for i in range(0, n_nodes-1):
        # Si(xi) = f(xi)
        A[4*i][4*i] = 1
        b[4*i] = y_vec[i]

        # Si(xi+1) = f(xi+1)
        h = x_vec[i+1] - x_vec[i]
        A[4*i + 1][4*i] = 1             # ai
        A[4*i + 1][4*i + 1] = h         # bi
        A[4*i + 1][4*i + 2] = h**2      # ci
        A[4*i + 1][4*i + 3] = h**3      # di
        b[4*i + 1] = y_vec[i+1]

        if i < n_nodes-2:
            # Si-1'(xi) = Si'(xi)
            A[4*i + 2][4*i] = 0
            A[4*i + 2][4*i + 1] = 1
            A[4*i + 2][4*i + 2] = 2 * h
            A[4*i + 2][4*i + 3] = 3 * (h**2)
            A[4*i + 2][4*(i+1)] = 0
            A[4*i + 2][4*(i+1) + 1] = -1
            b[4*i + 2] = 0

            # Si-1''(xi) = Si''(xi)
            A[4*i + 3][4*i] = 0
            A[4*i + 3][4*i + 1] = 0
            A[4*i + 3][4*i + 2] = 2
            A[4*i + 3][4*i + 3] = 6 * h
            A[4*i + 3][4*(i+1) + 2] = -2
            b[4*i + 3] = 0

    # S0''(x0) = 0
    A[4*(n_nodes-1) - 2][2] = 2
    b[4*(n_nodes-1) - 2] = 0

    # Sn-1''(xn) = 0
    h = x_vec[n_nodes - 1] - x_vec[n_nodes - 2]
    A[4*(n_nodes-1) - 1][4*(n_nodes-1) - 2] = 2
    A[4*(n_nodes-1) - 1][4*(n_nodes-1) - 1] = 6 * h
    b[4*(n_nodes-1) - 1] = 0

    x = np.linalg.solve(A, b)
    return x


def spline_interpolation(x_vec, y_vec, x_nodes, params):
    interpolated_values = np.zeros(len(x_vec))

    for i in range(len(x_vec)):
        for j in range(len(x_nodes)-1):
            if x_nodes[j] == x_vec[i]:
                interpolated_values[i] = y_vec[i]
                break
            elif x_nodes[j] <= x_vec[i] <= x_nodes[j+1]:
                a, b, c, d = params[4*j:4*j+4]
                h = x_vec[i] - x_nodes[j]
                interpolated_values[i] += a + b * h + c * (h**2) + d * (h**3)
                break
    interpolated_values[-1] = y_vec[-1]
    return interpolated_values


def plot_elevation_profile(x_vec, y_vec, interpolated_values, x_nodes, y_nodes, option, filename, location, n_nodes):
    method = "metodą Lagrange'a" if option == 'l' else "funkcjami sklejanymi"
    plot_title = f'Interpolacja {method}: {location}'
    plot_suptitle = f'Liczba węzłów: {n_nodes}'

    sns.set_style('whitegrid')
    colors = sns.color_palette('rocket')
    plt.figure(figsize=(8, 5))
    plt.plot(x_vec, y_vec, '.', label='dane', c=colors[1])
    plt.plot(x_vec, interpolated_values, label='funkcja interpolująca', c=colors[4])
    plt.plot(x_nodes, y_nodes, 'o', label='węzły interpolacji', color=colors[4])
    if max(interpolated_values) - max(y_vec) > 500 or abs(min(interpolated_values) - min(y_vec)) > 500:
        delta = max(y_vec) - min(y_vec)
        plt.ylim(min(y_vec) - 0.25*delta, max(y_vec) + 0.25*delta)
    plt.suptitle(plot_title, fontsize=14)
    plt.title(plot_suptitle, fontsize=10)
    plt.xlabel('Dystans [m]')
    plt.ylabel('Wysokość [m]')
    plt.legend()
    plt.savefig(filename)
    plt.show()


def estimate_elevation_profiles(x_data, y_data, k, filename, location):
    x_nodes = x_data[0::k]
    y_nodes = y_data[0::k]
    n_nodes = fit_data(x_data, y_data, x_nodes, y_nodes)

    fname_l = filename + '_lagrange' + str(n_nodes) + '.png'
    fname_s = filename + '_spline' + str(n_nodes) + '.png'

    # Lagrange
    interpolated_values = interpolation(x_nodes, y_nodes, x_data, n_nodes)
    plot_elevation_profile(x_data, y_data, interpolated_values, x_nodes, y_nodes, 'l', fname_l, location, n_nodes)

    # splines
    params = spline_parameters(x_nodes, y_nodes)
    interpolated_values = spline_interpolation(x_data, y_data, x_nodes, params)
    plot_elevation_profile(x_data, y_data, interpolated_values, x_nodes, y_nodes, 's', fname_s, location, n_nodes)


if __name__ == '__main__':
    x_data, y_data = read_txt_data("tczew_starogard.txt")
    estimate_elevation_profiles(x_data, y_data, 52, 'tczew_starogard', 'Trasa Tczew - Starogard Gdański')       # n_nodes = 11
    estimate_elevation_profiles(x_data, y_data, 32, 'tczew_starogard', 'Trasa Tczew - Starogard Gdański')       # n_nodes = 17
    estimate_elevation_profiles(x_data, y_data, 17, 'tczew_starogard', 'Trasa Tczew - Starogard Gdański')       # n_nodes = 32

    x_data, y_data = read_csv_data("WielkiKanionKolorado.csv")
    estimate_elevation_profiles(x_data, y_data, 52, 'kanion_kolorado', 'Wielki Kanion Kolorado')
    estimate_elevation_profiles(x_data, y_data, 32, 'kanion_kolorado', 'Wielki Kanion Kolorado')
    estimate_elevation_profiles(x_data, y_data, 17, 'kanion_kolorado', 'Wielki Kanion Kolorado')

    x_data, y_data = read_csv_data("MountEverest.csv")
    estimate_elevation_profiles(x_data, y_data, 52, 'mount_everest', 'Mount Everest')
    estimate_elevation_profiles(x_data, y_data, 32, 'mount_everest', 'Mount Everest')
    estimate_elevation_profiles(x_data, y_data, 17, 'mount_everest', 'Mount Everest')

