#ifndef RK4_H
#define RK4_H

#include <vector>

class RK4Integrator {
private:
    double dt;
    
public:
    RK4Integrator(double step_size) : dt(step_size) {}
    
    template<typename RHS>
    void step(double t, const std::vector<double>& y, RHS&& rhs,
              std::vector<double>& y_next) const {
        size_t n = y.size();
        std::vector<double> k1(n), k2(n), k3(n), k4(n), tmp(n);
        
        rhs(t, y, k1);
        for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + dt/2.0 * k1[i];
        rhs(t + dt/2.0, tmp, k2);
        for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + dt/2.0 * k2[i];
        rhs(t + dt/2.0, tmp, k3);
        for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + dt * k3[i];
        rhs(t + dt, tmp, k4);
        
        for (size_t i = 0; i < n; ++i)
            y_next[i] = y[i] + dt/6.0 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
    }
    
    void setDt(double new_dt) { dt = new_dt; }
    double getDt() const { return dt; }
};

#endif
