#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define ITERATIONS 100

static void matrix_vector_product(const std::vector<double>& a,
                                  const std::vector<double>& b,
                                  std::vector<double>& c,
                                  size_t m,
                                  size_t n)
{
    for (size_t i = 0; i < m; ++i) {
        c[i] = 0.0;
        for (size_t j = 0; j < n; ++j)
            c[i] += a[i * n + j] * b[j];
    }
}

static void init_matrix_sequential(std::vector<double>& a, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            a[i * n + j] = static_cast<double>(i + j);
}

static void init_vector_sequential(std::vector<double>& b, size_t n)
{
    for (size_t j = 0; j < n; ++j)
        b[j] = static_cast<double>(j);
}

static void init_matrix_parallel(std::vector<double>& a, size_t m, size_t n, int num_threads)
{
    std::vector<std::jthread> threads;
    threads.reserve(static_cast<size_t>(num_threads));
    for (int t = 0; t < num_threads; ++t) {
        const size_t row_begin = t * m / static_cast<size_t>(num_threads);
        const size_t row_end = (t + 1) * m / static_cast<size_t>(num_threads);
        threads.emplace_back([&, row_begin, row_end]() {
            for (size_t i = row_begin; i < row_end; ++i)
                for (size_t j = 0; j < n; ++j)
                    a[i * n + j] = static_cast<double>(i + j);
        });
    }
}

static void init_vector_parallel(std::vector<double>& b, size_t n, int num_threads)
{
    std::vector<std::jthread> threads;
    threads.reserve(static_cast<size_t>(num_threads));
    for (int t = 0; t < num_threads; ++t) {
        const size_t j_begin = t * n / static_cast<size_t>(num_threads);
        const size_t j_end = (t + 1) * n / static_cast<size_t>(num_threads);
        threads.emplace_back([&, j_begin, j_end]() {
            for (size_t j = j_begin; j < j_end; ++j)
                b[j] = static_cast<double>(j);
        });
    }
}

static void matrix_vector_product_parallel(const std::vector<double>& a,
                                           const std::vector<double>& b,
                                           std::vector<double>& c,
                                           size_t m,
                                           size_t n,
                                           int num_threads)
{
    std::vector<std::jthread> threads;
    threads.reserve(static_cast<size_t>(num_threads));
    for (int t = 0; t < num_threads; ++t) {
        const size_t row_begin = t * m / static_cast<size_t>(num_threads);
        const size_t row_end = (t + 1) * m / static_cast<size_t>(num_threads);
        threads.emplace_back([&, row_begin, row_end]() {
            for (size_t i = row_begin; i < row_end; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < n; ++j)
                    sum += a[i * n + j] * b[j];
                c[i] = sum;
            }
        });
    }
}

/** Как в 02/task 1: сначала инициализация, в замер входит только умножение. */
static double run_serial(size_t m, size_t n)
{
    std::vector<double> a(m * n);
    std::vector<double> b(n);
    std::vector<double> c(m);

    init_matrix_sequential(a, m, n);
    init_vector_sequential(b, n);

    const auto start = std::chrono::steady_clock::now();
    matrix_vector_product(a, b, c, m, n);
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

/**
 * Параллельная инициализация A и B: матрица и вектор заполняются одновременно (два jthread,
 * внутри каждого — рабочие jthread). Замер — только параллельное умножение (как в 02/task 1).
 */
static double run_parallel(size_t m, size_t n, int num_threads)
{
    std::vector<double> a(m * n);
    std::vector<double> b(n);
    std::vector<double> c(m);

    {
        std::jthread t_a([&] { init_matrix_parallel(a, m, n, num_threads); });
        std::jthread t_b([&] { init_vector_parallel(b, n, num_threads); });
    }

    const auto start = std::chrono::steady_clock::now();
    matrix_vector_product_parallel(a, b, c, m, n, num_threads);
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

static const std::vector<int> num_threads_list = {1, 2, 4, 6, 8};

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

    std::vector<double> t_serial_list;
    std::vector<double> t_parallel_list;

    std::cout << "Iterations: " << ITERATIONS << std::endl;
    for (int num_threads : num_threads_list) {
        double t_serial_sum = 0;
        double t_parallel_sum = 0;
        std::cout << "Running for num_threads: " << num_threads << " with M = " << M
                  << " and N = " << N << std::endl;
        for (int i = 0; i < ITERATIONS; i++) {
            const double t_serial = run_serial(M, N);
            const double t_parallel = run_parallel(M, N, num_threads);
            t_serial_sum += t_serial;
            t_parallel_sum += t_parallel;
            iteration_file << num_threads << "," << (i + 1) << "," << t_serial << "," << t_parallel
                           << "\n";
        }
        t_serial_list.push_back(t_serial_sum / ITERATIONS);
        t_parallel_list.push_back(t_parallel_sum / ITERATIONS);
    }

    for (size_t i = 0; i < num_threads_list.size(); i++) {
        const double speedup = t_serial_list[i] / t_parallel_list[i];
        summary_file << num_threads_list[i] << "," << t_serial_list[i] << "," << t_parallel_list[i]
                     << "," << speedup << "\n";
    }

    return 0;
}
