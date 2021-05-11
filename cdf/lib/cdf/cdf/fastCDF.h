// Copyright (C) 2020 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef FASTCDF_H
#define FASTCDF_H


/** \file fastCDF.h
 *  \brief Calculate CDF on a grid of points as explained in  Langrené-Warin 2020 :
 *         "Fast multivariate empirical cumulative distribution function with connection to kernel density estimation"
 *  \author Xavier Warin
 */


namespace StOpt
{
/// \brief Calculate fast CDF
/// \param p_x particules (sample)  size : (dimension, nbSim)
/// \param p_z  rectilinear points (in each dimension: points coordinates)
/// \param p_y  estimate for each p_x point
/// \return for each point of the grid return the CDF
Eigen::ArrayXd fastCDF(const Eigen::ArrayXXd &p_x, const std::vector< std::shared_ptr<Eigen::ArrayXd> >    &p_z, const Eigen::ArrayXd &p_y);

}
#endif /* FASTCDF_H */


