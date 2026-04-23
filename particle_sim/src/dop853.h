#ifndef DOP853_H
#define DOP853_H
#include <vector>
#include <cmath>
#include <algorithm>

class DOP853Integrator {
public:
    explicit DOP853Integrator(double h0,
                               double rtol    = 1e-8,
                               double atol    = 1e-10,
                               double h_min   = 1e-12,
                               double h_max   = 1.0,
                               int    max_rej = 100)
        : h_(h0), rtol_(rtol), atol_(atol),
          h_min_(h_min), h_max_(h_max), max_rej_(max_rej),
          fsal_valid_(false), n_accept_(0), n_reject_(0)
    {}

    template<typename RHS>
    bool step(double& t, std::vector<double>& y, RHS&& rhs) {
        const std::size_t n = y.size();
        _resize(n);

        if (!fsal_valid_) {
            rhs(t, y, k_[0]);
            fsal_valid_ = true;
        }

        int rej = 0;
        while (true) {
            double h = std::clamp(h_, h_min_, h_max_);

            _stages(rhs, t, y, h, n);

            for (std::size_t i = 0; i < n; ++i)
                y8_[i] = y[i] + h * (
                    B1  * k_[0][i]  + B6  * k_[5][i]  + B7  * k_[6][i] +
                    B8  * k_[7][i]  + B9  * k_[8][i]  + B10 * k_[9][i] +
                    B11 * k_[10][i] + B12 * k_[11][i]);

            for (std::size_t i = 0; i < n; ++i)
                err_[i] = h * (
                    E3_0  * k_[0][i]  + E3_6  * k_[5][i]  + E3_7  * k_[6][i] +
                    E3_8  * k_[7][i]  + E3_9  * k_[8][i]  + E3_10 * k_[9][i] +
                    E3_11 * k_[10][i] + E3_12 * k_[11][i]);

            double err = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double sc = atol_ + rtol_ * std::max(std::abs(y[i]),
                                                      std::abs(y8_[i]));
                double e  = err_[i] / sc;
                err += e * e;
            }
            err = std::sqrt(err / static_cast<double>(n));

            constexpr double safety  = 0.9;
            constexpr double fac_min = 0.333;
            constexpr double fac_max = 6.0;
            double fac = (err > 1e-300)
                         ? std::clamp(safety * std::pow(err, -1.0 / 8.0),
                                      fac_min, fac_max)
                         : fac_max;

            if (err <= 1.0) {
                t += h;
                y  = y8_;
                std::swap(k_[0], k_[12]);
                fsal_valid_ = true;
                h_ = std::clamp(h * fac, h_min_, h_max_);
                ++n_accept_;
                return true;
            }

            ++n_reject_;
            h_ = std::clamp(h * fac, h_min_, h_max_);
            rhs(t, y, k_[0]);

            if (++rej > max_rej_) {
                t += h;
                y  = y8_;
                std::swap(k_[0], k_[12]);
                fsal_valid_ = true;
                return false;
            }
        }
    }

    double getDt()   const { return h_; }
    void   setDt(double h) { h_ = h; fsal_valid_ = false; }
    void   setTolerances(double rt, double at) { rtol_ = rt; atol_ = at; }
    int    nAccept() const { return n_accept_; }
    int    nReject() const { return n_reject_; }

private:
    static constexpr double C2  = 0.526001519587677318e-01;
    static constexpr double C3  = 0.789002279381515978e-01;
    static constexpr double C4  = 0.118350341907227397e+00;
    static constexpr double C5  = 0.281649658092772603e+00;
    static constexpr double C6  = 0.333333333333333333e+00;
    static constexpr double C7  = 0.25e+00;
    static constexpr double C8  = 0.307692307692307692e+00;
    static constexpr double C9  = 0.651282051282051282e+00;
    static constexpr double C10 = 0.6e+00;
    static constexpr double C11 = 0.857142857142857142e+00;

    static constexpr double A21 =  5.26001519587677318e-02;

    static constexpr double A31 =  1.97250569845378994e-02;
    static constexpr double A32 =  5.91751709536136983e-02;

    static constexpr double A41 =  2.95875854768068491e-02;
    static constexpr double A43 =  8.87627564304205475e-02;

    static constexpr double A51 =  2.41365134159266680e-01;
    static constexpr double A53 = -8.84549479328286086e-01;
    static constexpr double A54 =  9.24834003261792003e-01;

    static constexpr double A61 =  3.70370370370370370e-02;
    static constexpr double A64 =  1.70828608729473871e-01;
    static constexpr double A65 =  1.25467687566822429e-01;

    static constexpr double A71 =  3.71093750000000000e-02;
    static constexpr double A74 =  1.70252211019544040e-01;
    static constexpr double A75 =  6.02165389804559092e-02;
    static constexpr double A76 = -1.75781250000000000e-02;

    static constexpr double A81 =  3.70920001185047927e-02;
    static constexpr double A84 =  1.70383925712239993e-01;
    static constexpr double A85 =  1.07262030446373284e-01;
    static constexpr double A86 = -1.53194377486244882e-02;
    static constexpr double A87 =  8.27378916381402268e-03;

    static constexpr double A91  =  6.24110958716075718e-01;
    static constexpr double A94  = -3.36089262944694141e+00;
    static constexpr double A95  = -8.68219346841726006e-01;
    static constexpr double A96  =  2.75920996994467132e+01;
    static constexpr double A97  =  2.01540675504778941e+01;
    static constexpr double A98  = -4.34898841810699588e+01;

    static constexpr double A10_1 =  4.77662536438264366e-01;
    static constexpr double A10_4 = -2.48811461997166764e+00;
    static constexpr double A10_5 = -5.90290826836842996e-01;
    static constexpr double A10_6 =  2.12300514481811942e+01;
    static constexpr double A10_7 =  1.52792336328824230e+01;
    static constexpr double A10_8 = -3.32882109689848600e+01;
    static constexpr double A10_9 = -2.03312017085086274e-02;

    static constexpr double A11_1  = -9.37142430085987273e-01;
    static constexpr double A11_4  =  5.18637242884406423e+00;
    static constexpr double A11_5  =  1.09143734899672952e+00;
    static constexpr double A11_6  = -8.14978701074692745e+00;
    static constexpr double A11_7  = -1.85200656599969586e+01;
    static constexpr double A11_8  =  2.27394870993505054e+01;
    static constexpr double A11_9  =  2.49360555267965234e+00;
    static constexpr double A11_10 = -3.04676447189821961e+00;

    static constexpr double A12_1  =  2.27331014751653800e+00;
    static constexpr double A12_4  = -1.05344954667372510e+01;
    static constexpr double A12_5  = -2.00087205822486247e+00;
    static constexpr double A12_6  = -1.79589318631188000e+01;
    static constexpr double A12_7  =  2.79488845294199600e+01;
    static constexpr double A12_8  = -2.85899827713502355e+00;
    static constexpr double A12_9  = -8.87285693353062953e+00;
    static constexpr double A12_10 =  1.23605671757943030e+01;
    static constexpr double A12_11 =  6.43392746015763600e-01;

    static constexpr double B1  =  5.42937341165687296e-02;
    static constexpr double B6  =  4.45031289275240888e+00;
    static constexpr double B7  =  1.89151789931450038e+00;
    static constexpr double B8  = -5.80120396001058470e+00;
    static constexpr double B9  =  3.11164366957819890e-01;
    static constexpr double B10 = -1.52160949662516078e-01;
    static constexpr double B11 =  2.01365400804030348e-01;
    static constexpr double B12 =  4.47106157277725905e-02;

    static constexpr double E3_0  = -1.89800754072407615e-01;
    static constexpr double E3_6  =  4.45031289275240888e+00;
    static constexpr double E3_7  =  1.89151789931450038e+00;
    static constexpr double E3_8  = -5.80120396001058470e+00;
    static constexpr double E3_9  = -4.22682321323791940e-01;
    static constexpr double E3_10 = -1.52160949662516078e-01;
    static constexpr double E3_11 =  2.01365400804030348e-01;
    static constexpr double E3_12 =  2.26517921983608200e-02;

    double h_, rtol_, atol_, h_min_, h_max_;
    int    max_rej_;
    bool   fsal_valid_;
    int    n_accept_, n_reject_;

    std::vector<double>              tmp_, y8_, err_;
    std::vector<std::vector<double>> k_;

    void _resize(std::size_t n) {
        if (tmp_.size() == n) return;
        tmp_.assign(n, 0.0);
        y8_ .assign(n, 0.0);
        err_.assign(n, 0.0);
        k_.assign(13, std::vector<double>(n, 0.0));
    }

    template<typename RHS>
    void _stages(RHS&& rhs, double t, const std::vector<double>& y,
                 double h, std::size_t n) {

        auto eval = [&](int s, double c,
                        std::initializer_list<std::pair<int,double>> row) {
            for (std::size_t i = 0; i < n; ++i) {
                tmp_[i] = y[i];
                for (auto [j, a] : row)
                    tmp_[i] += h * a * k_[j][i];
            }
            rhs(t + c * h, tmp_, k_[s]);
        };

        eval( 1, C2,  {{0, A21}});
        eval( 2, C3,  {{0, A31},  {1, A32}});
        eval( 3, C4,  {{0, A41},  {2, A43}});
        eval( 4, C5,  {{0, A51},  {2, A53},  {3, A54}});
        eval( 5, C6,  {{0, A61},  {3, A64},  {4, A65}});
        eval( 6, C7,  {{0, A71},  {3, A74},  {4, A75},  {5, A76}});
        eval( 7, C8,  {{0, A81},  {3, A84},  {4, A85},  {5, A86},  {6, A87}});
        eval( 8, C9,  {{0, A91},  {3, A94},  {4, A95},  {5, A96},  {6, A97},  {7, A98}});
        eval( 9, C10, {{0, A10_1},{3, A10_4},{4, A10_5},{5, A10_6},{6, A10_7},{7, A10_8},{8, A10_9}});
        eval(10, C11, {{0, A11_1},{3, A11_4},{4, A11_5},{5, A11_6},
                       {6, A11_7},{7, A11_8},{8, A11_9},{9, A11_10}});
        eval(11, 1.0, {{0, A12_1},{3, A12_4},{4, A12_5},{5, A12_6},
                       {6, A12_7},{7, A12_8},{8, A12_9},{9, A12_10},{10, A12_11}});

        for (std::size_t i = 0; i < n; ++i)
            tmp_[i] = y[i] + h * (
                B1  * k_[0][i]  + B6  * k_[5][i]  + B7  * k_[6][i] +
                B8  * k_[7][i]  + B9  * k_[8][i]  + B10 * k_[9][i] +
                B11 * k_[10][i] + B12 * k_[11][i]);
        rhs(t + h, tmp_, k_[12]);
    }
};

#endif // DOP853_H
