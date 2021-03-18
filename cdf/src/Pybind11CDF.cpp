// Copyright (C) 2020 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)

/** \file PyBind11StOptCDF.cpp
 * \brief Map CDF calculation   classes to python
 * \author Xavier Warin
 */

#include <Eigen/Dense>
#include <cdf/Pybind11VectorAndList.h>
#include <cdf/fastCDF.h>
#include <cdf/fastCDFOnSample.h>
#include <iostream>
#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace Eigen;

ArrayXd fastCDFWrap(const Eigen::ArrayXXd &p_x, const pybind11::list &p_z,
                    const Eigen::ArrayXd &p_y)
{
  std::vector<std::shared_ptr<Eigen::ArrayXd>> vecGrid(
      convertFromListShPtr<Eigen::ArrayXd>(p_z));

  return StOpt::fastCDF(p_x, vecGrid, p_y);
}

/// \brief Encapsulation for CDF module
PYBIND11_MODULE(cdf, m)
{
  m.def("fastCDF", &fastCDFWrap);
  m.def("fastCDFOnSample", &StOpt::fastCDFOnSample);
}
