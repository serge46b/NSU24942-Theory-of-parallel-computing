#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

#define ITERATIONS 5


const double epsilon = 1e-13;
const double tau = -1e-5;
const int N = 7100;


struct ScheduleConfig {
    omp_sched_t kind;
    int chunk;
    const char* name;
};


void solve_serial(const std::vector<double>& a,
                  const std::vector<double>& b,
                  std::vector<double>& x,
                  size_t n,
                  unsigned int num_threads)
{
    std::vector<double> diff(n);
    double b_len_sq = 0;
    double dev = 0;
    for (size_t i = 0; i < n; i++) {
        b_len_sq += b[i] * b[i];
        diff[i] = 0.0;
    }
    dev = b_len_sq;
    while (sqrt(dev / b_len_sq) >= epsilon) {
        dev = 0;
        for (size_t i = 0; i < n; ++i) {
            x[i] = x[i] - tau * diff[i];
        }
        for (size_t i = 0; i < n; ++i) {    
            double sum_ax = 0.0;
            for (size_t j = 0; j < n; ++j)
                sum_ax += a[i * n + j] * x[j];
            diff[i] = b[i] - sum_ax;
            dev += diff[i] * diff[i];
        }
    }
}

void solve_parallel_1(const std::vector<double>& a,
                    const std::vector<double>& b,
                    std::vector<double>& x,
                    size_t n,
                    unsigned int num_threads){
    std::vector<double> diff(n);
    double b_len_sq = 0.0;
    double dev = 0.0;
    for (size_t i = 0; i < n; ++i) {
        b_len_sq += b[i] * b[i];
        diff[i] = 0.0;
    }
    dev = b_len_sq;
    while (std::sqrt(dev / b_len_sq) >= epsilon) {
        dev = 0;
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < n; ++i) {
            x[i] = x[i] - tau * diff[i];
        }
        #pragma omp parallel for num_threads(num_threads) reduction(+:dev)
        for (size_t i = 0; i < n; ++i) {
            double sum_ax = 0.0;
            for (size_t j = 0; j < n; ++j)
                sum_ax += a[i * n + j] * x[j];
            diff[i] = b[i] - sum_ax;
            dev += diff[i] * diff[i];
        }
    }
}

void solve_parallel_2(const std::vector<double>& a,
    const std::vector<double>& b,
    std::vector<double>& x,
    size_t n,
    unsigned int num_threads)
{
    std::vector<double> diff(n);
    double b_len_sq = 0.0;

    for (size_t i = 0; i < n; ++i) {
        b_len_sq += b[i] * b[i];
        diff[i] = 0.0;
    }

    double dev = 0.0;
    bool stop = false;
    #pragma omp parallel num_threads(num_threads) shared(a,b,x,diff,dev,stop) firstprivate(b_len_sq)
    {
        while (!stop) {
            #pragma omp single
            {
                dev = 0.0;
            }
            #pragma omp for schedule(runtime)
            for (size_t i = 0; i < n; ++i) {
                x[i] = x[i] - tau * diff[i];
            }
            #pragma omp for schedule(runtime) reduction(+:dev)
            for (size_t i = 0; i < n; ++i) {
                double sum_ax = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    sum_ax += a[i * n + j] * x[j];
                }
                diff[i] = b[i] - sum_ax;
                dev += diff[i] * diff[i];
            }
            #pragma omp single
            {
                stop = (std::sqrt(dev / b_len_sq) < epsilon);
            }
        }
    }
}


double run_function(void (*solve)(const std::vector<double>&, const std::vector<double>&, std::vector<double>&, size_t, unsigned int),
                     size_t n, unsigned int num_threads){
    std::vector<double> a(n * n);
    std::vector<double> b(n);
    std::vector<double> x(n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            a[i * n + j] = 1.0 + (i==j ? 1.0 : 0.0);
        }
        b[i] = static_cast<double>(n+1);
        x[i] = 0.0;
    }
    const auto start = std::chrono::steady_clock::now();
    solve(a, b, x, n, num_threads);
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    // for (size_t i = 0; i < n; i++) {
    //     std::cout << "x[" << i << "] = " << x[i] << std::endl;
    // }
    return elapsed.count();
}

const std::vector<int> num_threads_list = {1,2,4,7,8,16,20,40};
const size_t ntl_index = 5;
const std::vector<ScheduleConfig> configs = {
    { omp_sched_static,  1,  "static,1" },
    { omp_sched_static,  4,  "static,4" },
    { omp_sched_static,  100,  "static,100" },
    { omp_sched_static,  N/num_threads_list[ntl_index],  "static,N/T" },
    { omp_sched_dynamic, 1,  "dynamic,1" },
    { omp_sched_dynamic, 4,  "dynamic,4" },
    { omp_sched_dynamic, 100,  "dynamic,100" },
    { omp_sched_dynamic, N/num_threads_list[ntl_index],  "dynamic,N/T" },
    { omp_sched_guided,  1,  "guided,1" },
    { omp_sched_guided, 4,  "guided,4" },
    { omp_sched_guided, 100,  "guided,100" },
    { omp_sched_guided, N/num_threads_list[ntl_index],  "guided,N/T" },
};

int main(void){
    // Phase 1: benchmark functions vs number of threads (similar to task 1 and task 2)
    std::ofstream iteration_file("iterationData.csv");
    std::ofstream summary_file("summary.csv");
    if (!iteration_file || !summary_file) {
        std::cerr << "Error: could not open output CSV files iterationData/summary.\n";
        return 1;
    }

    iteration_file << "num_threads,i,time_serial,time_parallel_1,time_parallel_2\n";
    summary_file << "num_threads,time_serial,time_parallel_1,time_parallel_2,speedup_1,speedup_2\n";

    double t_serial_sum = 0.0;
    double t_parallel1_sum = 0.0;
    double t_parallel2_sum = 0.0;
    std::vector<double> t_serial_list;
    std::vector<double> t_parallel1_list;
    std::vector<double> t_parallel2_list;

    std::cout << "Iterations (phase 1): " << ITERATIONS << std::endl;
    for (int num_threads : num_threads_list) {
        t_serial_sum = 0.0;
        t_parallel1_sum = 0.0;
        t_parallel2_sum = 0.0;
        std::cout << "Running phase 1 for num_threads: " << num_threads << std::endl;
        for (int i = 0; i < ITERATIONS; ++i) {
            std::cout << "Iteration " << i + 1 << " of " << ITERATIONS << std::endl;
            std::cout << "Running serial solution" << std::endl;
            double t_serial = run_function(solve_serial, N, num_threads);
            std::cout << "Running parallel solution 1" << std::endl;
            double t_parallel1 = run_function(solve_parallel_1, N, num_threads);
            std::cout << "Running parallel solution 2" << std::endl;
            double t_parallel2 = run_function(solve_parallel_2, N, num_threads);
            t_serial_sum += t_serial;
            t_parallel1_sum += t_parallel1;
            t_parallel2_sum += t_parallel2;
            iteration_file << num_threads << "," << (i + 1) << ","
                           << t_serial << "," << t_parallel1 << "," << t_parallel2 << "\n";
        }
        t_serial_list.push_back(t_serial_sum / ITERATIONS);
        t_parallel1_list.push_back(t_parallel1_sum / ITERATIONS);
        t_parallel2_list.push_back(t_parallel2_sum / ITERATIONS);
    }

    for (size_t i = 0; i < num_threads_list.size(); ++i) {
        double speedup1 = t_serial_list[i] / t_parallel1_list[i];
        double speedup2 = t_serial_list[i] / t_parallel2_list[i];
        summary_file << num_threads_list[i] << ","
                     << t_serial_list[i] << ","
                     << t_parallel1_list[i] << ","
                     << t_parallel2_list[i] << ","
                     << speedup1 << ","
                     << speedup2 << "\n";
    }

    // Phase 2: benchmark different schedule configurations using ntl_index
    std::ofstream iteration_file_sc("iterationData_sc.csv");
    std::ofstream summary_file_sc("summary_sc.csv");
    if (!iteration_file_sc || !summary_file_sc) {
        std::cerr << "Error: could not open output CSV files iterationData_sc/summary_sc.\n";
        return 1;
    }

    iteration_file_sc << "config_description,time_serial,time_parallel\n";
    summary_file_sc << "config_description,time_serial,time_parallel,speedup\n";

    const unsigned int num_threads_sc = static_cast<unsigned int>(num_threads_list[ntl_index]);
    std::cout << "Using num_threads_sc = " << num_threads_sc << " for schedule configs (phase 2)" << std::endl;

    for (const auto& cfg : configs) {
        omp_set_schedule(cfg.kind, cfg.chunk);

        double t_serial_sum_sc = 0.0;
        double t_parallel_sum_sc = 0.0;

        std::cout << "Running phase 2 for config: " << cfg.name
                  << " (kind=" << static_cast<int>(cfg.kind)
                  << ", chunk=" << cfg.chunk << ")\n";

        for (int i = 0; i < ITERATIONS; ++i) {
            double t_serial = run_function(solve_serial, N, num_threads_sc);
            double t_parallel = run_function(solve_parallel_2, N, num_threads_sc);
            t_serial_sum_sc += t_serial;
            t_parallel_sum_sc += t_parallel;

            iteration_file_sc << cfg.name << ","
                              << t_serial << ","
                              << t_parallel << "\n";
        }

        double t_serial_avg_sc = t_serial_sum_sc / ITERATIONS;
        double t_parallel_avg_sc = t_parallel_sum_sc / ITERATIONS;
        double speedup_sc = t_serial_avg_sc / t_parallel_avg_sc;
        summary_file_sc << cfg.name << ","
                        << t_serial_avg_sc << ","
                        << t_parallel_avg_sc << ","
                        << speedup_sc << "\n";
    }

    return 0;
}