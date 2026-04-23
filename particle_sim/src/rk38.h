#ifndef RK38_H
#define RK38_H

#include <vector>

class RK38Integrator {
private:
    double dt;
    
public:
    RK38Integrator(double step_size) : dt(step_size) {}
    
    template<typename RHS>
    void step(double t, const std::vector<double>& y, RHS&& rhs,
              std::vector<double>& y_next) const {
        size_t n = y.size();
        std::vector<double> k1(n), k2(n), k3(n), k4(n), tmp(n);
        
        rhs(t, y, k1);
        for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + dt/3.0 * k1[i];
        rhs(t + dt/3.0, tmp, k2);
        for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + dt * (-k1[i]/3.0 + k2[i]);
        rhs(t + 2.0*dt/3.0, tmp, k3);
        for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + dt * (k1[i] - k2[i] + k3[i]);
        rhs(t + dt, tmp, k4);
        
        for (size_t i = 0; i < n; ++i)
            y_next[i] = y[i] + dt/8.0 * (k1[i] + 3.0*k2[i] + 3.0*k3[i] + k4[i]);
    }
    
    void setDt(double new_dt) { dt = new_dt; }
    double getDt() const { return dt; }
};

#endif
