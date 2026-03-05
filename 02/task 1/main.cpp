#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

#define ITERATIONS 10



// Compute matrix-vector product c[m] = a[m][n] * b[n]
void matrix_vector_product(const std::vector<double>& a,
                           const std::vector<double>& b,
                           std::vector<double>& c,
                           size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i) {
        c[i] = 0.0;
        for (size_t j = 0; j < n; ++j)
            c[i] += a[i * n + j] * b[j];
    }
}

// Compute matrix-vector product c[m] = a[m][n] * b[n] (OpenMP parallel)
void matrix_vector_product_omp(const std::vector<double>& a,
                               const std::vector<double>& b,
                               std::vector<double>& c,
                               size_t m, size_t n, int num_threads)
{
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < static_cast<int>(m); ++i) {
        c[i] = 0.0;
        for (size_t j = 0; j < n; ++j)
            c[i] += a[i * n + j] * b[j];
    }
}

double run_serial(size_t m, size_t n)
{
    std::vector<double> a(m * n);
    std::vector<double> b(n);
    std::vector<double> c(m);

    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            a[i * n + j] = static_cast<double>(i + j);

    for (size_t j = 0; j < n; ++j)
        b[j] = static_cast<double>(j);

    const auto start = std::chrono::steady_clock::now();
    matrix_vector_product(a, b, c, m, n);
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    // std::cout << "Elapsed time (serial): " << elapsed.count() << " sec.\n";
    return elapsed.count();
}

double run_parallel(size_t m, size_t n, int num_threads)
{
    std::vector<double> a(m * n);
    std::vector<double> b(n);
    std::vector<double> c(m);

    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            a[i * n + j] = static_cast<double>(i + j);

    for (size_t j = 0; j < n; ++j)
        b[j] = static_cast<double>(j);

    const auto start = std::chrono::steady_clock::now();
    matrix_vector_product_omp(a, b, c, m, n, num_threads);
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;

    // std::cout << "Elapsed time (parallel): " << elapsed.count() << " sec.\n";
    return elapsed.count();
}

const std::vector<int> num_threads_list = {1,2,4,7,8,16,20,40};

int main(int argc, char* argv[])
{
    size_t M = 1000;
    size_t N = 1000;

    if (argc > 1) {
        try {
            M = std::stoul(argv[1]);
        } catch (const std::exception&) {
            std::cerr << "Invalid M, using default 1000\n";
        }
    }
    if (argc > 2) {
        try {
            N = std::stoul(argv[2]);
        } catch (const std::exception&) {
            std::cerr << "Invalid N, using default 1000\n";
        }
    }

    std::ofstream iteration_file("iterationData.csv");
    std::ofstream summary_file("summary.csv");
    if (!iteration_file || !summary_file) {
        std::cerr << "Error: could not open output CSV files.\n";
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
        std::cout << "Running for num_threads: " << num_threads << " with M = " << M << " and N = " << N << std::endl;
        for (int i = 0; i < ITERATIONS; i++) {
            double t_serial = run_serial(M, N);
            double t_parallel = run_parallel(M, N, num_threads);
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
