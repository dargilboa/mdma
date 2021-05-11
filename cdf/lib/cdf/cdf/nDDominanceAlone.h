#ifndef _NDDOMINANCEALONE_H
#define _NDDOMINANCEALONE_H
#include <vector>
#include <memory>
#include <Eigen/Dense>


/** _file nDDominanceAlone.h
 *  \brief using dominance method as in Bouchard Warin 2009, and Langrene Warin 2020, calculate CDF
 * \author Xavier Warin
 */
namespace StOpt
{
/// \brief  Dominance  calculation
///         \f$   \sum_{j=1}^N valToAdd(j) 1_{pt(j) < pt(i)}  \forall i \f$
/// \param  p_pt         N dimension points (ndim, nb sample)
/// \param  p_valToAdd   value to add
void nDDominanceAlone(const Eigen::ArrayXXd &p_pt,
                      const Eigen::ArrayXd    &p_valToAdd,
                      Eigen::ArrayXd   &p_fDomin);
}

#endif
