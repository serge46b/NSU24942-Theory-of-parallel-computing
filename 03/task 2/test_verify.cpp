#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {
constexpr double eps = 1e-12;

bool skip_line(const std::string& s)
{
    return s.empty() || s[0] == '#';
}

bool near(double a, double b)
{
    return std::abs(a - b) <= eps * (1.0 + std::max(std::abs(a), std::abs(b)));
}

int check_sin(const char* path)
{
    std::ifstream in(path);
    if (!in) {
        std::cerr << "missing " << path << "\n";
        return 1;
    }
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
        if (skip_line(line))
            continue;
        std::size_t id;
        double x, r;
        std::istringstream iss(line);
        if (!(iss >> id >> x >> r)) {
            std::cerr << "bad line in " << path << ": " << line << "\n";
            return 1;
        }
        if (!near(r, std::sin(x))) {
            std::cerr << "sin mismatch id=" << id << " x=" << x << " got=" << r << " exp=" << std::sin(x) << "\n";
            return 1;
        }
        ++n;
    }
    std::cout << path << ": " << n << " ok\n";
    return 0;
}

int check_sqrt(const char* path)
{
    std::ifstream in(path);
    if (!in) {
        std::cerr << "missing " << path << "\n";
        return 1;
    }
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
        if (skip_line(line))
            continue;
        std::size_t id;
        double x, r;
        std::istringstream iss(line);
        if (!(iss >> id >> x >> r)) {
            std::cerr << "bad line in " << path << ": " << line << "\n";
            return 1;
        }
        if (!near(r, std::sqrt(x))) {
            std::cerr << "sqrt mismatch id=" << id << "\n";
            return 1;
        }
        ++n;
    }
    std::cout << path << ": " << n << " ok\n";
    return 0;
}

int check_pow(const char* path)
{
    std::ifstream in(path);
    if (!in) {
        std::cerr << "missing " << path << "\n";
        return 1;
    }
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
        if (skip_line(line))
            continue;
        std::size_t id;
        double b, e, r;
        std::istringstream iss(line);
        if (!(iss >> id >> b >> e >> r)) {
            std::cerr << "bad line in " << path << ": " << line << "\n";
            return 1;
        }
        if (!near(r, std::pow(b, e))) {
            std::cerr << "pow mismatch id=" << id << " b=" << b << " e=" << e << "\n";
            return 1;
        }
        ++n;
    }
    std::cout << path << ": " << n << " ok\n";
    return 0;
}
} // namespace

int main()
{
    int e = 0;
    e |= check_sin("client1_sin.txt");
    e |= check_sqrt("client2_sqrt.txt");
    e |= check_pow("client3_pow.txt");
    return e;
}
