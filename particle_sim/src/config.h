#ifndef CONFIG_H
#define CONFIG_H

struct SystemParams {
    double E0 = 0.5;      // амплитуда электрического поля
    double B0 = 1.0;      // амплитуда магнитного поля
    double omega = 1.0;   // частота полей
    double omega_c = B0;  // циклотронная частота (q=1, m=1)
};

struct InitialConditions {
    double x = 1.0;
    double y = 0.0;
    double z = 0.0;
    double vx = 0.0;
    double vy = 0.5;
    double vz = 0.2;
};

struct IntegrationParams {
    double t_start = 0.0;
    double t_end = 50.0;
    double dt = 0.001;
};

#endif
