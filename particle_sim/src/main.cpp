#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <sys/stat.h>

#include "rk4.h"
#include "rk38.h"
#include "rk56.h"
#include "dopri5.h"
#include "dop853.h"
#include "particle.h"
#include "config.h"

#ifdef _WIN32
#include <direct.h>
#define mkdir(dir, mode) _mkdir(dir)
#endif

class RHSWrapper {
    const Particle& particle;
public:
    RHSWrapper(const Particle& p) : particle(p) {}
    void operator()(double t,
                    const std::vector<double>& y,
                    std::vector<double>& dydt) const {
        particle.rhs(t, y, dydt);
    }
};

class FieldWrapper {
    const Particle& particle;
public:
    FieldWrapper(const Particle& p) : particle(p) {}
    void operator()(double t,
                    double& Ex, double& Ey, double& Ez,
                    double& Bx, double& By, double& Bz) const {
        particle.computeFields(t, Ex, Ey, Ez, Bx, By, Bz);
    }
};

void ensureDirectory(const std::string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) != 0)
        mkdir(dir.c_str(), 0777);
}

void writeCSVLine(std::ofstream& file,
                  double t,
                  const std::vector<double>& y,
                  double Ex, double Ey, double Ez,
                  double Bx, double By, double Bz,
                  double dt)
{
    file << std::scientific << std::setprecision(12)
         << t   << ","
         << y[0] << "," << y[1] << "," << y[2] << ","
         << y[3] << "," << y[4] << "," << y[5] << ","
         << Ex << "," << Ey << "," << Ez << ","
         << Bx << "," << By << "," << Bz << ","
         << dt << "\n";
}

template<typename Integrator>
void runFixed(double omega, double dt, double t_end,
              const std::string& filename,
              const std::string& name)
{
    std::cout << name << " : omega=" << omega
              << ", dt=" << dt << std::endl;

    SystemParams params;
    params.omega = omega;
    Particle particle(params);

    RHSWrapper rhs(particle);
    FieldWrapper field(particle);

    std::vector<double> y = {1.0, 0.0, 0.0, 0.0, 0.5, 0.2};
    std::vector<double> y_new(6);
    double t = 0.0;

    Integrator integrator(dt);

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    file << "t,x,y,z,vx,vy,vz,Ex,Ey,Ez,Bx,By,Bz,dt\n";

    std::size_t n_steps = static_cast<std::size_t>(t_end / dt);

    for (std::size_t i = 0; i <= n_steps; ++i) {
        double Ex, Ey, Ez, Bx, By, Bz;
        field(t, Ex, Ey, Ez, Bx, By, Bz);
        writeCSVLine(file, t, y, Ex, Ey, Ez, Bx, By, Bz, dt);

        if (i < n_steps) {
            integrator.step(t, y, rhs, y_new);
            y = y_new;
            t = static_cast<double>(i + 1) * dt;
        }
    }

    file.close();
    std::cout << "  Saved to: " << filename << std::endl;
}

void runAdaptiveDOPRI5(double omega, double dt0, double t_end,
                       const std::string& filename,
                       double rel_tol, double abs_tol,
                       const std::string& name)
{
    std::cout << name << " : omega=" << omega
              << ", dt0=" << dt0
              << ", rtol=" << rel_tol << ", atol=" << abs_tol << std::endl;

    SystemParams params;
    params.omega = omega;
    Particle particle(params);

    RHSWrapper rhs(particle);
    FieldWrapper field(particle);

    std::vector<double> y = {1.0, 0.0, 0.0, 0.0, 0.5, 0.2};
    double t = 0.0;

    DOPRI5Integrator integrator(dt0, rel_tol, abs_tol);

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    file << "t,x,y,z,vx,vy,vz,Ex,Ey,Ez,Bx,By,Bz,dt\n";

    int step_count = 0;
    const int max_steps = 1'000'000;
    double dt_next = dt0;

    while (t < t_end && step_count < max_steps) {
        double Ex, Ey, Ez, Bx, By, Bz;
        field(t, Ex, Ey, Ez, Bx, By, Bz);
        writeCSVLine(file, t, y, Ex, Ey, Ez, Bx, By, Bz, integrator.getDt());

        integrator.step(t, y, rhs, dt_next);
        ++step_count;
    }

    file.close();
    std::cout << "  Steps: " << step_count
              << ", final t=" << t
              << ", final dt=" << integrator.getDt() << std::endl;
    std::cout << "  Saved to: " << filename << std::endl;
}

void runAdaptiveRK56(double omega, double dt0, double t_end,
                     const std::string& filename,
                     double tol,
                     const std::string& name)
{
    std::cout << name << " : omega=" << omega
              << ", dt0=" << dt0 << ", tol=" << tol << std::endl;

    SystemParams params;
    params.omega = omega;
    Particle particle(params);

    RHSWrapper rhs(particle);
    FieldWrapper field(particle);

    std::vector<double> y = {1.0, 0.0, 0.0, 0.0, 0.5, 0.2};
    double t = 0.0;

    RK56Integrator integrator(dt0, tol);

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    file << "t,x,y,z,vx,vy,vz,Ex,Ey,Ez,Bx,By,Bz,dt\n";

    int step_count = 0;
    const int max_steps = 1'000'000;

    while (t < t_end && step_count < max_steps) {
        double Ex, Ey, Ez, Bx, By, Bz;
        field(t, Ex, Ey, Ez, Bx, By, Bz);
        writeCSVLine(file, t, y, Ex, Ey, Ez, Bx, By, Bz, integrator.getDt());

        integrator.step(t, y, rhs);
        ++step_count;
    }

    file.close();
    std::cout << "  Steps: " << step_count
              << ", final t=" << t
              << ", final dt=" << integrator.getDt() << std::endl;
    std::cout << "  Saved to: " << filename << std::endl;
}

void runAdaptiveDOP853(double omega, double dt0, double t_end,
                       const std::string& filename,
                       double rel_tol, double abs_tol,
                       const std::string& name)
{
    std::cout << name << " : omega=" << omega
              << ", dt0=" << dt0
              << ", rtol=" << rel_tol << ", atol=" << abs_tol << std::endl;

    SystemParams params;
    params.omega = omega;
    Particle particle(params);

    RHSWrapper rhs(particle);
    FieldWrapper field(particle);

    std::vector<double> y = {1.0, 0.0, 0.0, 0.0, 0.5, 0.2};
    double t = 0.0;

    DOP853Integrator integrator(dt0, rel_tol, abs_tol);

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    file << "t,x,y,z,vx,vy,vz,Ex,Ey,Ez,Bx,By,Bz,dt\n";

    int step_count = 0;
    const int max_steps = 1'000'000;

    while (t < t_end && step_count < max_steps) {
        if (t + integrator.getDt() > t_end)
            integrator.setDt(t_end - t);

        double Ex, Ey, Ez, Bx, By, Bz;
        field(t, Ex, Ey, Ez, Bx, By, Bz);
        writeCSVLine(file, t, y, Ex, Ey, Ez, Bx, By, Bz, integrator.getDt());

        bool ok = integrator.step(t, y, rhs);
        if (!ok)
            std::cerr << "  Warning: DOP853 step rejected at t=" << t << std::endl;

        ++step_count;
    }

    file.close();
    std::cout << "  Steps: "    << step_count
              << "  (accept="   << integrator.nAccept()
              << ", reject="    << integrator.nReject() << ")"
              << ", final t="   << t
              << ", final dt="  << integrator.getDt() << std::endl;
    std::cout << "  Saved to: " << filename << std::endl;
}

int main() {
    ensureDirectory("output");
    ensureDirectory("plots");

    double omega, dt;
    int method_choice;

    std::cout << "Enter omega: ";
    std::cin >> omega;
    std::cout << "Enter initial dt: ";
    std::cin >> dt;
    std::cout << "\n";
    
    std::cout << "Select method:\n";
    std::cout << "1. RK4\n";
    std::cout << "2. RK3/8\n";
    std::cout << "3. RK5(6)\n";
    std::cout << "4. DOPRI5\n";
    std::cout << "5. DOP853\n";
    std::cout << "Enter number: ";
    std::cin >> method_choice;
    std::cout << "\n";

    const double t_end = 50.0;
    
    switch(method_choice) {
        case 1:
            runFixed<RK4Integrator>(omega, dt, t_end, "output/rk4.csv", "RK4");
            break;
        case 2:
            runFixed<RK38Integrator>(omega, dt, t_end, "output/rk38.csv", "RK3/8");
            break;
        case 3:
            runAdaptiveRK56(omega, dt, t_end, "output/rk56.csv", 1e-8, "RK5(6)");
            break;
        case 4:
            runAdaptiveDOPRI5(omega, dt, t_end, "output/dopri5.csv", 1e-8, 1e-10, "DOPRI5");
            break;
        case 5:
            runAdaptiveDOP853(omega, dt, t_end, "output/dop853.csv", 1e-10, 1e-12, "DOP853");
            break;
        default:
            std::cerr << "Invalid choice\n";
            runFixed<RK4Integrator>(omega, dt, t_end, "output/rk4.csv", "RK4");
            break;
    }

    return 0;
}
