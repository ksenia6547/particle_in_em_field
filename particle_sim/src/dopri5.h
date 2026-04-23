#ifndef DOPRI5_H
#define DOPRI5_H

#include <vector>
#include <cmath>
#include <algorithm>

class DOPRI5Integrator {
private:
    double dt;
    double dt_min;
    double dt_max;
    double safety;
    double facmin;
    double facmax;
    double rtol;
    double atol;
    
public:
    DOPRI5Integrator(double step_size, 
                     double rel_tol = 1e-6, 
                     double abs_tol = 1e-8,
                     double min_step = 1e-12,
                     double max_step = 1.0) 
        : dt(step_size), dt_min(min_step), dt_max(max_step),
          safety(0.9), facmin(0.2), facmax(10.0),
          rtol(rel_tol), atol(abs_tol) {}
    
    template<typename RHS>
    bool step(double& t, std::vector<double>& y, RHS&& rhs,
              double& dt_next) {
        size_t n = y.size();
        double h = dt;
        
        const double c2 = 1.0/5.0;
        const double c3 = 3.0/10.0;
        const double c4 = 4.0/5.0;
        const double c5 = 8.0/9.0;
        const double c6 = 1.0;
        const double c7 = 1.0;
        
        const double a21 = 1.0/5.0;
        const double a31 = 3.0/40.0, a32 = 9.0/40.0;
        const double a41 = 44.0/45.0, a42 = -56.0/15.0, a43 = 32.0/9.0;
        const double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0, a53 = 64448.0/6561.0, a54 = -212.0/729.0;
        const double a61 = 9017.0/3168.0, a62 = -355.0/33.0, a63 = 46732.0/5247.0, a64 = 49.0/176.0, a65 = -5103.0/18656.0;
        const double a71 = 35.0/384.0, a72 = 0.0, a73 = 500.0/1113.0, a74 = 125.0/192.0, a75 = -2187.0/6784.0, a76 = 11.0/84.0;
        
        const double b1 = 35.0/384.0, b2 = 0.0, b3 = 500.0/1113.0,
                     b4 = 125.0/192.0, b5 = -2187.0/6784.0, b6 = 11.0/84.0, b7 = 0.0;
        
        const double b1_star = 5179.0/57600.0, b2_star = 0.0, b3_star = 7571.0/16695.0,
                     b4_star = 393.0/640.0, b5_star = -92097.0/339200.0, 
                     b6_star = 187.0/2100.0, b7_star = 1.0/40.0;
        
        std::vector<std::vector<double>> k(7, std::vector<double>(n));
        std::vector<double> tmp(n);
        std::vector<double> y_next(n);
        std::vector<double> y_4th(n);
        
        bool step_accepted = false;
        int attempts = 0;
        const int max_attempts = 100;
        
        while (!step_accepted && attempts < max_attempts) {
            attempts++;
            h = std::min(h, dt_max);
            
            rhs(t, y, k[0]);
            
            for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + h * a21 * k[0][i];
            rhs(t + c2*h, tmp, k[1]);
            
            for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + h * (a31*k[0][i] + a32*k[1][i]);
            rhs(t + c3*h, tmp, k[2]);
            
            for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + h * (a41*k[0][i] + a42*k[1][i] + a43*k[2][i]);
            rhs(t + c4*h, tmp, k[3]);
            
            for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + h * (a51*k[0][i] + a52*k[1][i] + a53*k[2][i] + a54*k[3][i]);
            rhs(t + c5*h, tmp, k[4]);
            
            for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + h * (a61*k[0][i] + a62*k[1][i] + a63*k[2][i] + a64*k[3][i] + a65*k[4][i]);
            rhs(t + c6*h, tmp, k[5]);
            
            for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + h * (a71*k[0][i] + a72*k[1][i] + a73*k[2][i] + a74*k[3][i] + a75*k[4][i] + a76*k[5][i]);
            rhs(t + c7*h, tmp, k[6]);
            
            for (size_t i = 0; i < n; ++i) {
                y_next[i] = y[i] + h * (b1*k[0][i] + b2*k[1][i] + b3*k[2][i] +
                                        b4*k[3][i] + b5*k[4][i] + b6*k[5][i] + b7*k[6][i]);
                
                y_4th[i] = y[i] + h * (b1_star*k[0][i] + b2_star*k[1][i] + b3_star*k[2][i] +
                                       b4_star*k[3][i] + b5_star*k[4][i] + b6_star*k[5][i] + b7_star*k[6][i]);
            }
            
            double error_norm = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double scale = atol + rtol * std::max(std::abs(y[i]), std::abs(y_next[i]));
                double err = std::abs(y_next[i] - y_4th[i]) / scale;
                error_norm += err * err;
            }
            error_norm = std::sqrt(error_norm / n);
            
            if (error_norm <= 1.0) {
                step_accepted = true;
                y = y_next;
                t = t + h;
                
                double fac = safety * std::pow(error_norm, -0.2);
                fac = std::max(facmin, std::min(facmax, fac));
                dt_next = h * fac;
                dt_next = std::min(dt_max, std::max(dt_min, dt_next));
                dt = dt_next;
            } else {
                double fac = safety * std::pow(error_norm, -0.25);
                fac = std::max(facmin, std::min(facmax, fac));
                h = h * fac;
                h = std::min(dt_max, std::max(dt_min, h));
                
                if (h < dt_min) {
                    y = y_next;
                    t = t + h;
                    dt_next = h;
                    dt = dt_next;
                    step_accepted = true;
                }
            }
        }
        
        return step_accepted;
    }
    
    template<typename RHS>
    void step_fixed(double t, const std::vector<double>& y, RHS&& rhs,
                    std::vector<double>& y_next) const {
        size_t n = y.size();
        const double h = dt;
        
        const double c2 = 1.0/5.0, c3 = 3.0/10.0, c4 = 4.0/5.0, c5 = 8.0/9.0, c6 = 1.0, c7 = 1.0;
        const double a21 = 1.0/5.0;
        const double a31 = 3.0/40.0, a32 = 9.0/40.0;
        const double a41 = 44.0/45.0, a42 = -56.0/15.0, a43 = 32.0/9.0;
        const double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0, a53 = 64448.0/6561.0, a54 = -212.0/729.0;
        const double a61 = 9017.0/3168.0, a62 = -355.0/33.0, a63 = 46732.0/5247.0, a64 = 49.0/176.0, a65 = -5103.0/18656.0;
        const double a71 = 35.0/384.0, a72 = 0.0, a73 = 500.0/1113.0, a74 = 125.0/192.0, a75 = -2187.0/6784.0, a76 = 11.0/84.0;
        const double b1 = 35.0/384.0, b2 = 0.0, b3 = 500.0/1113.0,
                     b4 = 125.0/192.0, b5 = -2187.0/6784.0, b6 = 11.0/84.0, b7 = 0.0;
        
        std::vector<std::vector<double>> k(7, std::vector<double>(n));
        std::vector<double> tmp(n);
        
        rhs(t, y, k[0]);
        for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + h * a21 * k[0][i];
        rhs(t + c2*h, tmp, k[1]);
        for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + h * (a31*k[0][i] + a32*k[1][i]);
        rhs(t + c3*h, tmp, k[2]);
        for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + h * (a41*k[0][i] + a42*k[1][i] + a43*k[2][i]);
        rhs(t + c4*h, tmp, k[3]);
        for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + h * (a51*k[0][i] + a52*k[1][i] + a53*k[2][i] + a54*k[3][i]);
        rhs(t + c5*h, tmp, k[4]);
        for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + h * (a61*k[0][i] + a62*k[1][i] + a63*k[2][i] + a64*k[3][i] + a65*k[4][i]);
        rhs(t + c6*h, tmp, k[5]);
        for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + h * (a71*k[0][i] + a72*k[1][i] + a73*k[2][i] + a74*k[3][i] + a75*k[4][i] + a76*k[5][i]);
        rhs(t + c7*h, tmp, k[6]);
        
        for (size_t i = 0; i < n; ++i) {
            y_next[i] = y[i] + h * (b1*k[0][i] + b2*k[1][i] + b3*k[2][i] +
                                    b4*k[3][i] + b5*k[4][i] + b6*k[5][i] + b7*k[6][i]);
        }
    }
    
    void setDt(double new_dt) { dt = new_dt; }
    double getDt() const { return dt; }
    void setTolerances(double rel_tol, double abs_tol) { rtol = rel_tol; atol = abs_tol; }
};

#endif
