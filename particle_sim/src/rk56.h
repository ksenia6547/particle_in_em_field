#ifndef RK56_ADAPTIVE_H
#define RK56_ADAPTIVE_H

#include <vector>
#include <cmath>
#include <algorithm>

class RK56Integrator {
public:
    RK56Integrator(double step_size,
                   double tolerance = 1e-6,
                   double dt_min_   = 1e-10,
                   double dt_max_   = 1.0)
        : dt(step_size), rtol(tolerance), atol(tolerance * 1e-3),
          dt_min(dt_min_), dt_max(dt_max_) {}

    template<typename RHS>
    void step(double& t, std::vector<double>& y, RHS&& rhs) {
        const size_t n = y.size();

        std::vector<std::vector<double>> k(6, std::vector<double>(n));
        std::vector<double> y_tmp(n), y5(n), y4(n);

        // Узлы
        static constexpr double c2 = 1.0/4.0;
        static constexpr double c3 = 3.0/8.0;
        static constexpr double c4 = 12.0/13.0;
        static constexpr double c5 = 1.0;
        static constexpr double c6 = 1.0/2.0;

        auto eval = [&](int s, double c,
                        std::initializer_list<std::pair<int,double>> row,
                        double h) {
            for (size_t i = 0; i < n; ++i) {
                y_tmp[i] = y[i];
                for (auto& [j, a] : row)
                    y_tmp[i] += h * a * k[j][i];
            }
            rhs(t + c * h, y_tmp, k[s]);
        };

        int rej = 0;
        while (true) {
            double h = std::clamp(dt, dt_min, dt_max);

            rhs(t, y, k[0]);
            eval(1, c2, {{0, 1.0/4.0}}, h);
            eval(2, c3, {{0, 3.0/32.0},      {1, 9.0/32.0}}, h);
            eval(3, c4, {{0, 1932.0/2197.0},  {1, -7200.0/2197.0}, {2, 7296.0/2197.0}}, h);
            eval(4, c5, {{0, 439.0/216.0},    {1, -8.0},
                          {2, 3680.0/513.0},   {3, -845.0/4104.0}}, h);
            eval(5, c6, {{0, -8.0/27.0},      {1, 2.0},
                          {2, -3544.0/2565.0}, {3, 1859.0/4104.0}, {4, -11.0/40.0}}, h);

            for (size_t i = 0; i < n; ++i)
                y5[i] = y[i] + h * (
                     16.0/135.0    * k[0][i] +
                   6656.0/12825.0  * k[2][i] +
                  28561.0/56430.0  * k[3][i] +
                     -9.0/50.0     * k[4][i] +
                      2.0/55.0     * k[5][i]);

            for (size_t i = 0; i < n; ++i)
                y4[i] = y[i] + h * (
                     25.0/216.0   * k[0][i] +
                   1408.0/2565.0  * k[2][i] +
                   2197.0/4104.0  * k[3][i] +
                     -1.0/5.0     * k[4][i]);

            double err = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double sc = atol + rtol * std::max(std::abs(y[i]),
                                                    std::abs(y5[i]));
                double e  = (y5[i] - y4[i]) / sc;
                err += e * e;
            }
            err = std::sqrt(err / static_cast<double>(n));

            constexpr double safety  = 0.9;
            constexpr double fac_min = 0.2;
            constexpr double fac_max = 5.0;
            double fac = (err > 1e-300)
                         ? std::clamp(safety * std::pow(err, -1.0/5.0),
                                      fac_min, fac_max)
                         : fac_max;

            if (err <= 1.0) {
                t  += h;
                y   = y5;
                dt  = std::clamp(h * fac, dt_min, dt_max);
                return;
            }

            dt = std::clamp(h * fac, dt_min, dt_max);
            ++rej;

            if (dt <= dt_min) {
                t += h;
                y  = y5;
                return;
            }
        }
    }

    double getDt() const { return dt; }
    void setDt(double new_dt) { dt = new_dt; }

private:
    double dt, rtol, atol, dt_min, dt_max;
};

#endif
