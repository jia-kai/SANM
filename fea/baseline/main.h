#include <Eigen/Core>

namespace baseline {
using CoordMat = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using IndexMat = Eigen::Matrix<int, 4, Eigen::Dynamic>;
using MaskMat = Eigen::Matrix<bool, 3, Eigen::Dynamic>;

struct Stat {
    int nr_iter = 0, nr_iter_refine = 0;
    double tot_time = 0, tot_newton_time = 0, df = 0, dx = 0, grad_rms = 0,
           grad_rms_refine = 0, energy = 0;
    CoordMat vtx;  //!< solved vertex locations
};

struct MaterialDesc {
    enum Energy {
        NEOHOOKEAN_C,
        NEOHOOKEAN_I,
        ARAP,
    };
    Energy energy;
    double young, poisson;
};

//! energy minimization with optional static external force
Stat solve_energy_min(const IndexMat& elements, const CoordMat& vtx_init,
                      const CoordMat& vtx_dst, const CoordMat* f_ext,
                      const MaskMat& bnd_mask,
                      const MaterialDesc& material_desc, double gtol_refine);

//! solve static force equilibrium with the Levenberg-Marquardt algorithm
Stat solve_force_equ_levmar(const IndexMat& elements, const CoordMat& vtx_init,
                            const CoordMat& f_ext, const MaskMat& bnd_mask,
                            const MaterialDesc& material_desc, double gtol);

extern bool g_hessian_proj;
extern double g_hessian_diag_reg;
}  // namespace baseline
