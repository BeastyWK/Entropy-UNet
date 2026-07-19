/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "pair_ljlambda.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "utils.h"

using namespace LAMMPS_NS;
using namespace MathConst;

// 物理常数（与 3SPN.2 一致，确保单位制兼容）
#define _KB_ 1.3806505E-23
#define _NA_ 6.0221415E23
#define _EC_ 1.60217653E-19
#define _PV_ 8.8541878176E-22

/* ---------------------------------------------------------------------- */

PairLJLambda::PairLJLambda(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairLJLambda::~PairLJLambda()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut_lj);
    memory->destroy(cut_ljsq);
    memory->destroy(cut_coul);
    memory->destroy(cut_coulsq);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
    memory->destroy(lambda);
  }
}

/* ---------------------------------------------------------------------- */

void PairLJLambda::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double qtmp, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, ecoul, fpair;
  double rsq, r2inv, r6inv, forcecoul, forcelj, factor_coul, factor_lj;
  int *ilist, *jlist, *numneigh, **firstneigh;
  double r, rinv, screening;

  double TWO_1_3 = pow(2.0, (1.0/3.0));

  evdwl = ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag, vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0 / rsq;

        // --- 屏蔽库仑（静电）---
        if (rsq < cut_coulsq[itype][jtype]) {
          r = sqrt(rsq);
          rinv = 1.0 / r;
          screening = exp(-kappa * r);
          forcecoul = qqrd2e * qtmp * q[j] * screening * (kappa + rinv);
        } else {
          forcecoul = 0.0;
        }

        // --- Lennard-Jones（带 lambda 缩放）---
        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv = r2inv * r2inv * r2inv;
          if (rsq <= TWO_1_3 * sigma[itype][jtype] * sigma[itype][jtype]) {
            forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
          } else {
            forcelj = lambda[itype][jtype] * r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
          }
        } else {
          forcelj = 0.0;
        }

        fpair = (factor_coul * forcecoul + factor_lj * forcelj) * r2inv;

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

        if (eflag) {
          if (rsq < cut_coulsq[itype][jtype]) {
            ecoul = factor_coul * qqrd2e * qtmp * q[j] * rinv * screening;
          } else {
            ecoul = 0.0;
          }
          if (rsq < cut_ljsq[itype][jtype]) {
            if (rsq <= TWO_1_3 * sigma[itype][jtype] * sigma[itype][jtype]) {
              evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) + (1 - lambda[itype][jtype]) * epsilon[itype][jtype];
            } else {
              evdwl = lambda[itype][jtype] * r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
            }
            evdwl *= factor_lj;
          } else {
            evdwl = 0.0;
          }
        }

        if (evflag) {
          ev_tally(i, j, nlocal, newton_pair, evdwl, ecoul, fpair, delx, dely, delz);
        }
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJLambda::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

  memory->create(cut_lj, n + 1, n + 1, "pair:cut_lj");
  memory->create(cut_ljsq, n + 1, n + 1, "pair:cut_ljsq");
  memory->create(cut_coul, n + 1, n + 1, "pair:cut_coul");
  memory->create(cut_coulsq, n + 1, n + 1, "pair:cut_coulsq");
  memory->create(epsilon, n + 1, n + 1, "pair:epsilon");
  memory->create(sigma, n + 1, n + 1, "pair:sigma");
  memory->create(lj1, n + 1, n + 1, "pair:lj1");
  memory->create(lj2, n + 1, n + 1, "pair:lj2");
  memory->create(lj3, n + 1, n + 1, "pair:lj3");
  memory->create(lj4, n + 1, n + 1, "pair:lj4");
  memory->create(offset, n + 1, n + 1, "pair:offset");
  memory->create(lambda, n + 1, n + 1, "pair:lambda");
}

/* ----------------------------------------------------------------------
   全局设置：自动计算 kappa
   语法：pair_style ljlambda T salt_mM cut_lj [cut_coul]
   示例：pair_style ljlambda 300 100 12.0 35.0
   （注意：cut_coul 仅作为母版，实际会被 3.5/kappa 覆盖）
------------------------------------------------------------------------- */

void PairLJLambda::settings(int narg, char **arg)
{
  if (narg < 4 || narg > 5)
    error->all(FLERR, "Illegal pair_style command. Usage: pair_style ljlambda T salt_mM cut_lj [cut_coul]");

  double temp = utils::numeric(FLERR, arg[0], false, lmp);
  double salt_mM = utils::numeric(FLERR, arg[1], false, lmp);
  double salt_conc = salt_mM / 1000.0;   // 转换为 M (mol/L)

  if (salt_conc <= 0.0)
    error->all(FLERR, "Salt concentration must be positive");

  cut_lj_global = utils::numeric(FLERR, arg[2], false, lmp);
  if (narg == 4) {
    cut_coul_global = cut_lj_global;
    if (comm->me == 0) {
      utils::logmesg(lmp, "WARNING: cut_coul not specified in pair_style, defaulted to cut_lj = %g A.\n"
                         "This value will likely be overridden by pair_coeff or init_one.\n", cut_lj_global);
    }
  } else {
    cut_coul_global = utils::numeric(FLERR, arg[3], false, lmp);
  }

  // 计算介电常数（3SPN.2 非线性公式）
  double dielectric = 249.4 - 7.88E-01 * temp + 7.20E-04 * temp * temp;
  dielectric *= (1.000 - (0.2551 * salt_conc) +
                 0.05151 * salt_conc * salt_conc -
                 0.006889 * salt_conc * salt_conc * salt_conc);

  // 计算德拜长度（Å）
  double ldby = sqrt(dielectric * _PV_ * _KB_ * temp * 1.0E27 /
                     (2.0 * _NA_ * _EC_ * _EC_ * salt_conc));

  kappa = 1.0 / ldby;   // 逆德拜长度

  if (comm->me == 0) {
    utils::logmesg(lmp, "LJLambda Auto-DH: T = %g K, Salt = %g mM\n", temp, salt_mM);
    utils::logmesg(lmp, "              Dielectric = %g, Debye length = %g A, kappa = %g 1/A\n",
                   dielectric, ldby, kappa);
    utils::logmesg(lmp, "              Automatic Coulomb cutoff (3.5/kappa) = %g A\n", 3.5/kappa);
  }

  // 将全局截断应用于已设定的类型对
  if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i + 1; j <= atom->ntypes; j++)
        if (setflag[i][j]) {
          cut_lj[i][j] = cut_lj_global;
          cut_coul[i][j] = cut_coul_global;
        }
  }
}

/* ----------------------------------------------------------------------
   设置原子对系数
   语法（完整版）：pair_coeff i j ljlambda epsilon sigma lambda cut_lj cut_coul
   语法（简化版）：pair_coeff i j ljlambda epsilon sigma lambda cut_lj
                  （此时 cut_coul 自动 = 3.5 / kappa）
   当 cut_coul = 0.0 时，该原子对静电被强制关闭
------------------------------------------------------------------------- */

void PairLJLambda::coeff(int narg, char **arg)
{
  if (narg < 6 || narg > 7)
    error->all(FLERR, "Incorrect args. Usage: pair_coeff ... epsilon sigma lambda cut_lj [cut_coul]");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double epsilon_one = utils::numeric(FLERR, arg[2], false, lmp);
  double sigma_one = utils::numeric(FLERR, arg[3], false, lmp);
  double lambda_one = utils::numeric(FLERR, arg[4], false, lmp);
  double cut_lj_one = utils::numeric(FLERR, arg[5], false, lmp);

  double cut_coul_one;
  if (narg == 7) {
    // 用户显式写了一个值（可以是 0 来关闭静电，也可以是正数来覆盖）
    cut_coul_one = utils::numeric(FLERR, arg[6], false, lmp);
  } else {
    // 用户没写最后的 cut_coul → 自动采用 3.5 倍德拜长度
    if (kappa <= 0.0)
      error->all(FLERR, "kappa not initialized. Make sure pair_style ljlambda is called before pair_coeff.");
    cut_coul_one = 3.5 / kappa;
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      lambda[i][j] = lambda_one;
      cut_lj[i][j] = cut_lj_one;
      cut_coul[i][j] = cut_coul_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJLambda::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR, "Pair style ljlambda requires atom attribute q");
  neighbor->request(this, instance_me);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
   改进：未定义的对，库仑截断自动采用 3.5/kappa，避免母版污染
------------------------------------------------------------------------- */

double PairLJLambda::init_one(int i, int j)
{
  // 如果用户没有通过 pair_coeff 显式定义这对
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i], epsilon[j][j],
                               sigma[i][i], sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i], sigma[j][j]);
    cut_lj[i][j] = mix_distance(cut_lj[i][i], cut_lj[j][j]);

    // --- 核心改进：未定义的对，库仑截断自动按 3.5/kappa，不混合全局母版 ---
    if (kappa > 0.0) {
      cut_coul[i][j] = 3.5 / kappa;   // 自动最优截断
    } else {
      // 如果 kappa 尚未初始化（极少数情况），fallback 到混合
      cut_coul[i][j] = mix_distance(cut_coul[i][i], cut_coul[j][j]);
    }
  }

  double cut = MAX(cut_lj[i][j], cut_coul[i][j]);
  cut_ljsq[i][j] = cut_lj[i][j] * cut_lj[i][j];
  cut_coulsq[i][j] = cut_coul[i][j] * cut_coul[i][j];

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j], 12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j], 6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j], 12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j], 6.0);

  if (offset_flag) {
    double ratio = sigma[i][j] / cut_lj[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio, 12.0) - pow(ratio, 6.0));
  } else {
    offset[i][j] = 0.0;
  }

  cut_ljsq[j][i] = cut_ljsq[i][j];
  cut_coulsq[j][i] = cut_coulsq[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  lambda[j][i] = lambda[i][j];
  sigma[j][i] = sigma[i][j];
  epsilon[j][i] = epsilon[i][j];
  offset[j][i] = offset[i][j];

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;
    double count[2], all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count, all, 2, MPI_DOUBLE, MPI_SUM, world);
    double sig2 = sigma[i][j] * sigma[i][j];
    double sig6 = sig2 * sig2 * sig2;
    double rc3 = cut_lj[i][j] * cut_lj[i][j] * cut_lj[i][j];
    double rc6 = rc3 * rc3;
    double rc9 = rc3 * rc6;
    etail_ij = 8.0 * MY_PI * all[0] * all[1] * epsilon[i][j] *
               sig6 * (sig6 - 3.0 * rc6) / (9.0 * rc9);
    ptail_ij = 16.0 * MY_PI * all[0] * all[1] * epsilon[i][j] *
               sig6 * (2.0 * sig6 - 3.0 * rc6) / (9.0 * rc9);
  }
  return cut;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJLambda::write_restart(FILE *fp)
{
  write_restart_settings(fp);
  int i, j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j], sizeof(double), 1, fp);
        fwrite(&sigma[i][j], sizeof(double), 1, fp);
        fwrite(&lambda[i][j], sizeof(double), 1, fp);
        fwrite(&cut_lj[i][j], sizeof(double), 1, fp);
        fwrite(&cut_coul[i][j], sizeof(double), 1, fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJLambda::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();
  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, NULL, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR, &epsilon[i][j], sizeof(double), 1, fp, NULL, error);
          utils::sfread(FLERR, &sigma[i][j], sizeof(double), 1, fp, NULL, error);
          utils::sfread(FLERR, &lambda[i][j], sizeof(double), 1, fp, NULL, error);
          utils::sfread(FLERR, &cut_lj[i][j], sizeof(double), 1, fp, NULL, error);
          utils::sfread(FLERR, &cut_coul[i][j], sizeof(double), 1, fp, NULL, error);
        }
        MPI_Bcast(&epsilon[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&sigma[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&lambda[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut_lj[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut_coul[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJLambda::write_restart_settings(FILE *fp)
{
  fwrite(&cut_lj_global, sizeof(double), 1, fp);
  fwrite(&cut_coul_global, sizeof(double), 1, fp);
  fwrite(&kappa, sizeof(double), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
  fwrite(&tail_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJLambda::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR, &cut_lj_global, sizeof(double), 1, fp, NULL, error);
    utils::sfread(FLERR, &cut_coul_global, sizeof(double), 1, fp, NULL, error);
    utils::sfread(FLERR, &kappa, sizeof(double), 1, fp, NULL, error);
    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, NULL, error);
    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, NULL, error);
    utils::sfread(FLERR, &tail_flag, sizeof(int), 1, fp, NULL, error);
  }
  MPI_Bcast(&cut_lj_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&cut_coul_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&kappa, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&tail_flag, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLJLambda::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp, "%d %g %g\n", i, epsilon[i][i], sigma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLJLambda::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g\n", i, j, epsilon[i][j], sigma[i][j], cut_lj[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairLJLambda::single(int i, int j, int itype, int jtype,
                            double rsq,
                            double factor_coul, double factor_lj,
                            double &fforce)
{
  double r2inv, r6inv, forcecoul, forcelj, phicoul, philj;
  double r, rinv, screening;
  double TWO_1_3 = pow(2.0, (1.0/3.0));

  r2inv = 1.0 / rsq;
  if (rsq < cut_coulsq[itype][jtype]) {
    r = sqrt(rsq);
    rinv = 1.0 / r;
    screening = exp(-kappa * r);
    forcecoul = force->qqrd2e * atom->q[i] * atom->q[j] * screening * (kappa + rinv);
  } else {
    forcecoul = 0.0;
  }

  if (rsq < cut_ljsq[itype][jtype]) {
    r6inv = r2inv * r2inv * r2inv;
    if (rsq <= TWO_1_3 * sigma[itype][jtype] * sigma[itype][jtype]) {
      forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
    } else {
      forcelj = lambda[itype][jtype] * r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
    }
  } else {
    forcelj = 0.0;
  }

  fforce = (factor_coul * forcecoul + factor_lj * forcelj) * r2inv;

  double eng = 0.0;
  if (rsq < cut_coulsq[itype][jtype]) {
    phicoul = force->qqrd2e * atom->q[i] * atom->q[j] * rinv * screening;
    eng += factor_coul * phicoul;
  }
  if (rsq < cut_ljsq[itype][jtype]) {
    if (rsq <= TWO_1_3 * sigma[itype][jtype] * sigma[itype][jtype]) {
      philj = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) + (1 - lambda[itype][jtype]) * epsilon[itype][jtype];
    } else {
      philj = lambda[itype][jtype] * r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
    }
    eng += factor_lj * philj;
  }
  return eng;
}

/* ---------------------------------------------------------------------- */

void *PairLJLambda::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str, "cut_coul") == 0) return (void *) &cut_coul;
  dim = 2;
  if (strcmp(str, "epsilon") == 0) return (void *) epsilon;
  if (strcmp(str, "sigma") == 0) return (void *) sigma;
  if (strcmp(str, "lambda") == 0) return (void *) lambda;
  return NULL;
}