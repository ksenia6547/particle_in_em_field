#ifndef PARTICLE_H
#define PARTICLE_H

#include "config.h"
#include <cmath>
#include <vector>

class Particle {
private:
    SystemParams params;
    
public:
    Particle(const SystemParams& p) : params(p) {}
    
    void computeFields(double t, double& Ex, double& Ey, double& Ez,
                       double& Bx, double& By, double& Bz) const {
        Ex = params.E0 * std::cos(params.omega * t);
        Ey = -params.E0 * std::sin(params.omega * t);
        Ez = 0.0;

        Bx = 0.0;
        By = 0.0;
        Bz = params.B0;
    }
    
    void rhs(double t, const std::vector<double>& y, std::vector<double>& dydt) const {
        double vx = y[3], vy = y[4], vz = y[5];
        double Ex, Ey, Ez, Bx, By, Bz;
        computeFields(t, Ex, Ey, Ez, Bx, By, Bz);
        
        dydt[0] = vx;
        dydt[1] = vy;
        dydt[2] = vz;
        dydt[3] = Ex + (vy * Bz - vz * By);
        dydt[4] = Ey + (vz * Bx - vx * Bz);
        dydt[5] = Ez + (vx * By - vy * Bx);
    }
    
    const SystemParams& getParams() const { return params; }
    void setOmega(double newOmega) { params.omega = newOmega; }
};

#endif
