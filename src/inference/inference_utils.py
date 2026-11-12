"""Inference-only action helpers; action-rate guards remain with the callers."""
import numpy as np


def fill_in(actions):
    if len(actions.shape) > 2:
        actions = actions.squeeze(0) # Ensure actions is 2D
    new_action_list = np.zeros((actions.shape[0] * 2, actions.shape[1]), dtype=np.float32)
    for i in range(actions.shape[0]):
        action = actions[i]
        new_action_list[i* 2] = action
        if i < len(actions) - 1:
            fill_action = action / 2 + actions[i + 1] / 2
            new_action_list[i* 2 + 1] = fill_action
        else:
            #simply append the last action to make an even 100
            new_action_list[-1] = action

    # put back in np and reshape
    new_action_list = np.array(new_action_list)
    new_action_list = np.expand_dims(new_action_list, axis=0)  # Add batch dimension
    return new_action_list
