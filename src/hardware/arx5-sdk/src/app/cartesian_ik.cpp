#include "cartesian_ik.h"

#include <limits>
#include <stdexcept>

namespace arx
{
namespace detail
{

namespace
{
void validate_joint_vectors(const VecDoF &current_joint_position, const VecDoF &joint_pos_min,
                            const VecDoF &joint_pos_max)
{
    if (current_joint_position.size() == 0 || current_joint_position.size() != joint_pos_min.size() ||
        current_joint_position.size() != joint_pos_max.size())
    {
        throw std::invalid_argument("IK joint position and limit vectors must have equal nonzero sizes");
    }
}
} // namespace

bool is_valid_ik_candidate(const IKCandidate &candidate, const VecDoF &joint_pos_min,
                           const VecDoF &joint_pos_max)
{
    if (candidate.status != 0 || candidate.joint_position.size() != joint_pos_min.size() ||
        joint_pos_min.size() != joint_pos_max.size() || !candidate.joint_position.allFinite())
    {
        return false;
    }

    return ((candidate.joint_position - joint_pos_min).array() > 0.0).all() &&
           ((joint_pos_max - candidate.joint_position).array() > 0.0).all();
}

std::tuple<int, VecDoF> select_closest_valid_ik_candidate(const std::vector<IKCandidate> &candidates,
                                                           const VecDoF &current_joint_position,
                                                           const VecDoF &joint_pos_min,
                                                           const VecDoF &joint_pos_max)
{
    validate_joint_vectors(current_joint_position, joint_pos_min, joint_pos_max);
    if (candidates.empty())
        throw std::invalid_argument("IK candidate list must not be empty");

    const IKCandidate *closest = nullptr;
    double closest_distance = std::numeric_limits<double>::infinity();
    for (const IKCandidate &candidate : candidates)
    {
        if (!is_valid_ik_candidate(candidate, joint_pos_min, joint_pos_max))
            continue;

        const double distance = (candidate.joint_position - current_joint_position).squaredNorm();
        if (distance < closest_distance)
        {
            closest = &candidate;
            closest_distance = distance;
        }
    }

    if (closest != nullptr)
        return std::make_tuple(closest->status, closest->joint_position);

    // Holding the measured position is safer than commanding a failed,
    // partially converged, or out-of-limit IK result.
    const int failure_status = candidates.front().status == 0 ? -9 : candidates.front().status;
    return std::make_tuple(failure_status, current_joint_position);
}

} // namespace detail
} // namespace arx
