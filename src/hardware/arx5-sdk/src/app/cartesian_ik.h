#ifndef CARTESIAN_IK_H
#define CARTESIAN_IK_H

#include "app/common.h"

#include <tuple>
#include <vector>

namespace arx
{
namespace detail
{

struct IKCandidate
{
    int status;
    VecDoF joint_position;
};

bool is_valid_ik_candidate(const IKCandidate &candidate, const VecDoF &joint_pos_min,
                           const VecDoF &joint_pos_max);

std::tuple<int, VecDoF> select_closest_valid_ik_candidate(const std::vector<IKCandidate> &candidates,
                                                           const VecDoF &current_joint_position,
                                                           const VecDoF &joint_pos_min,
                                                           const VecDoF &joint_pos_max);

} // namespace detail
} // namespace arx

#endif // CARTESIAN_IK_H
