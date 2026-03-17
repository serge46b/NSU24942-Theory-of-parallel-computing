#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

#define ITERATIONS 100

const double PI = 3.14159265358979323846;
const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

double func(double x)
{
    return exp(-x * x);
}

double integrate(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));

    sum *= h;

    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n, int num_threads)
{
    double h = (b - a) / n;
    double sum = 0.0;

#pragma omp parallel for reduction(+:sum) num_threads(num_threads)
    for (int i = 0; i < n; ++i)
    {
        sum += func(a + h * (i + 0.5));
    }
    
    sum *= h;
    return sum;
}


double integrate_omp_atomic(double (*func)(double), double a, double b, int n, int num_threads)
{
    double h = (b - a) / n;
    double sum = 0.0;

#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; ++i)
    {
        double local_sum = func(a + h * (i + 0.5));
        #pragma omp atomic
            sum += local_sum;
    }
    
    sum *= h;
    return sum;
}


double run_serial()
{
    const auto start = std::chrono::steady_clock::now();
    double res = integrate(func, a, b, nsteps);
    const auto end = std::chrono::steady_clock::now();
    volatile double sink = res;  // prevent dead-code elimination
    (void)sink;
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

double run_parallel_atomic(int num_threads)
{
    const auto start = std::chrono::steady_clock::now();
    double res = integrate_omp_atomic(func, a, b, nsteps, num_threads);
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

void set_binding_policy(const std::string &policy)
{
  if (policy == "close")
  {
    setenv("OMP_PROC_BIND", "close", 1);
    setenv("OMP_PLACES", "cores", 1);
    std::cout << "Binding policy: CLOSE\n";
  }
  else if (policy == "spread")
  {
    setenv("OMP_PROC_BIND", "spread", 1);
    setenv("OMP_PLACES", "cores", 1);
    std::cout << "Binding policy: SPREAD\n";
  }
  else
  {
    setenv("OMP_PROC_BIND", "false", 1);
    std::cout << "Binding policy: NO BIND\n";
  }
}


const std::vector<int> num_threads_list = {1,2,4,7,8,16,20,40};
// const std::vector<int> num_threads_list = {1};
// const std::vector<int> num_threads_list = {1,2,4,8};

const std::vector<std::string> binding_policies = {"close", "spread", "false"};

int main()
{
    std::cout << "Integration f(x) on [" << a << ", " << b << "], nsteps = " << nsteps << std::endl;

    std::vector<std::ofstream> iteration_files;
    std::vector<std::ofstream> summary_files;
    for (const std::string &policy : binding_policies) {
        iteration_files.push_back(std::ofstream("iterationData_" + policy + ".csv"));
        summary_files.push_back(std::ofstream("summary_" + policy + ".csv"));
    }

    for (size_t policy_index = 0; policy_index < binding_policies.size(); policy_index++) {
        iteration_files[policy_index] << "num_threads,i,time_serial,time_parallel\n";
        summary_files[policy_index] << "num_threads,time_serial,time_parallel,time_parallel_atomic,speedup,speedup_atomic\n";
    }

    for (size_t policy_index = 0; policy_index < binding_policies.size(); policy_index++) {
        set_binding_policy(binding_policies[policy_index]);
        double t_serial_sum = 0;
        double t_parallel_sum = 0;
        double t_parallel_atomic_sum = 0;
        std::vector<double> t_serial_list;
        std::vector<double> t_parallel_list;
        std::vector<double> t_parallel_atomic_list;
        std::cout << "Iterations: " << ITERATIONS << std::endl;
        for (int num_threads : num_threads_list) {
            t_serial_sum = 0;
            t_parallel_sum = 0;
            t_parallel_atomic_sum = 0;
            std::cout << "Running for num_threads: " << num_threads << std::endl;
            for (int i = 0; i < ITERATIONS; i++) {
                double t_serial = run_serial();
                double t_parallel = run_parallel(num_threads);
                double t_parallel_atomic = run_parallel_atomic(num_threads);
                t_serial_sum += t_serial;
                t_parallel_sum += t_parallel;
                t_parallel_atomic_sum += t_parallel_atomic;
                iteration_files[policy_index] << num_threads << "," << (i + 1) << "," << t_serial << "," << t_parallel << "," << t_parallel_atomic << "\n";
            }
            t_serial_list.push_back(t_serial_sum / ITERATIONS);
            t_parallel_list.push_back(t_parallel_sum / ITERATIONS);
            t_parallel_atomic_list.push_back(t_parallel_atomic_sum / ITERATIONS);
        }

        for (size_t i = 0; i < num_threads_list.size(); i++) {
            double speedup = t_serial_list[i] / t_parallel_list[i];
            double speedup_atomic = t_serial_list[i] / t_parallel_atomic_list[i];
            summary_files[policy_index] << num_threads_list[i] << "," << t_serial_list[i] << "," << t_parallel_list[i] << "," << t_parallel_atomic_list[i] << "," << speedup << "," << speedup_atomic << "\n";
        }
    }

    return 0;
}
