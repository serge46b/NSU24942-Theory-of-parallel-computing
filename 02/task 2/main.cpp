#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

#define ITERATIONS 50

const double PI = 3.14159265358979323846;
const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

double func(double x)
{
    return exp(-x * x);
}

template <typename Func>
double integrate(Func func, double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));

    sum *= h;

    return sum;
}

template <typename Func>
double integrate_omp(Func func, double a, double b, int n, int num_threads)
{
    double h = (b - a) / n;
    double sum = 0.0;

    #pragma omp parallel for schedule(dynamic, 100) num_threads(num_threads) reduction(+ : sum)
        for (int i = 0; i < n; ++i)
        {
            sum += func(a + h * (i + 0.5));
        }
    sum *= h;

    return sum;
}

double run_serial()
{
    const auto start = std::chrono::steady_clock::now();
    double res = integrate(func, a, b, nsteps);
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}
double run_parallel(int num_threads)
{
    const auto start = std::chrono::steady_clock::now();
    double res = integrate_omp(func, a, b, nsteps, num_threads);
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

const std::vector<int> num_threads_list = {1,2,4,7,8,16,20,40};
// const std::vector<int> num_threads_list = {1,2,4,8};

int main()
{
    std::cout << "Integration f(x) on [" << a << ", " << b << "], nsteps = " << nsteps << std::endl;

    
    std::ofstream iteration_file("iterationData.csv");
    std::ofstream summary_file("summary.csv");
    if (!iteration_file || !summary_file) {
        std::cerr << "Error: could not open output CSV files." << std::endl;
        return 1;
    }

    iteration_file << "num_threads,i,time_serial,time_parallel\n";
    summary_file << "num_threads,time_serial,time_parallel,speedup\n";

    double t_serial_sum = 0;
    double t_parallel_sum = 0;
    std::vector<double> t_serial_list;
    std::vector<double> t_parallel_list;

    std::cout << "Iterations: " << ITERATIONS << std::endl;
    for (int num_threads : num_threads_list) {
        t_serial_sum = 0;
        t_parallel_sum = 0;
        std::cout << "Running for num_threads: " << num_threads << std::endl;
        for (int i = 0; i < ITERATIONS; i++) {
            double t_serial = run_serial();
            double t_parallel = run_parallel(num_threads);
            t_serial_sum += t_serial;
            t_parallel_sum += t_parallel;
            iteration_file << num_threads << "," << (i + 1) << "," << t_serial << "," << t_parallel << "\n";
        }
        t_serial_list.push_back(t_serial_sum / ITERATIONS);
        t_parallel_list.push_back(t_parallel_sum / ITERATIONS);
    }

    for (size_t i = 0; i < num_threads_list.size(); i++) {
        double speedup = t_serial_list[i] / t_parallel_list[i];
        summary_file << num_threads_list[i] << "," << t_serial_list[i] << "," << t_parallel_list[i] << "," << speedup << "\n";
    }

    return 0;
}
