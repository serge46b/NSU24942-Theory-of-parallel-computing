#include "server.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>

namespace {
constexpr int N = 500; // 5 < N < 10000
const char* const F1 = "client1_sin.txt";
const char* const F2 = "client2_sqrt.txt";
const char* const F3 = "client3_pow.txt";

thread_local std::mt19937 rng{std::random_device{}()};
} // namespace

int main()
{
    TaskServer<double> server(4);
    std::jthread server_thread([&] { server.start(); });
    {
        std::jthread c1([&] {
            std::ofstream out(F1);
            out << std::setprecision(17);
            out << "# id x sin(x)\n";
            std::uniform_real_distribution<double> ux(-6.283185307179586, 6.283185307179586);
            std::vector<std::size_t> ids;
            std::vector<double> xs;
            for (int i = 0; i < N; ++i) {
                const double x = ux(rng);
                const std::size_t id = server.add_task({TaskDesc::Sin, x, 0});
                ids.push_back(id);
                xs.push_back(x);
            }
            for (int i = 0; i < N; ++i) {
                const auto id = ids[i];
                const double r = server.request_result(id);
                out << id << ' ' << xs[i] << ' ' << r << '\n';
            }
        });

        std::jthread c2([&] {
            std::ofstream out(F2);
            out << std::setprecision(17);
            out << "# id x sqrt(x)\n";
            std::uniform_real_distribution<double> ux(0.0, 1e6);
            std::vector<std::size_t> ids;
            std::vector<double> xs;
            for (int i = 0; i < N; ++i) {
                const double x = ux(rng);
                const std::size_t id = server.add_task({TaskDesc::Sqrt, x, 0});
                ids.push_back(id);
                xs.push_back(x);
            }
            for (int i = 0; i < N; ++i) {
                const auto id = ids[i];
                const double r = server.request_result(id);
                out << id << ' ' << xs[i] << ' ' << r << '\n';
            }
        });

        std::jthread c3([&] {
            std::ofstream out(F3);
            out << std::setprecision(17);
            out << "# id base exp pow(base,exp)\n";
            std::uniform_real_distribution<double> ub(0.01, 20.0);
            std::uniform_real_distribution<double> ue(-5.0, 5.0);
            std::vector<std::size_t> ids;
            std::vector<double> bases;
            std::vector<double> exponents;
            for (int i = 0; i < N; ++i) {
                const double b = ub(rng);
                const double e = ue(rng);
                const std::size_t id = server.add_task({TaskDesc::Pow, b, e});
                ids.push_back(id);
                bases.push_back(b);
                exponents.push_back(e);            }
            for (int i = 0; i < N; ++i) {
                const auto id = ids[i];
                const double r = server.request_result(id);
                out << id << ' ' << bases[i] << ' ' << exponents[i] << ' ' << r << '\n';

            }
        });
    }
    server.stop();
    server_thread.join();
    std::cout << "done: " << F1 << ", " << F2 << ", " << F3 << " (" << N << " lines each)\n";
    return 0;
}
