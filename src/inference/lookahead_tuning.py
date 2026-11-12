"""Operator tuning of the direct-inference action and gripper lookahead while paused.

Two numbers, counted in actions: the action lookahead is the index at which each
chunk after the first starts (the actions before it are skipped at the chunk
boundary); the gripper lookahead is how many further actions ahead the gripper is
read. This module only holds and validates the numbers. The inference loop reads
them when a pause ends; it never changes them during policy motion.
"""

# The key listener lowercases keys, so arrows arrive as '\x1b[a' .. '\x1b[d'.
KEYS = {'\x1b[a': ('action', 1), 'w': ('action', 1), '\x1b[b': ('action', -1), 's': ('action', -1),
        '\x1b[c': ('gripper', 1), 'd': ('gripper', 1), '\x1b[d': ('gripper', -1), 'a': ('gripper', -1)}
HELP = 'Lookahead tuning while paused: up/down (w/s) action lookahead, right/left (d/a) gripper lookahead.'


class LookaheadTuner:
    def __init__(self, action, gripper, *, chunk_steps, horizon, fps, notify=print):
        self.action, self.gripper = int(action), int(gripper)
        self.chunk_steps, self.horizon, self.fps = int(chunk_steps), int(horizon), float(fps)
        self.notify = notify

    def __str__(self):
        ms = 1000.0 / self.fps
        return (f'[Lookahead] action {self.action} ({self.action * ms:.0f} ms), '
                f'gripper +{self.gripper} ({(self.action + self.gripper) * ms:.0f} ms)')

    def adjust(self, key):
        """Step one number for a lookahead key. Return whether `key` was a lookahead key.

        A value that would run past the end of a chunk (executed steps plus both
        lookaheads beyond the prediction horizon), or below zero, is refused.
        """
        if key not in KEYS:
            return False
        name, direction = KEYS[key]
        action = self.action + (direction if name == 'action' else 0)
        gripper = self.gripper + (direction if name == 'gripper' else 0)
        if min(action, gripper) < 0:
            self.notify(f'{self}: {name} lookahead cannot go below 0')
        elif self.chunk_steps + action + gripper > self.horizon:
            self.notify(f'{self}: refused, {self.chunk_steps} executed steps + {action} + {gripper} '
                        f'exceeds the {self.horizon}-action prediction horizon')
        else:
            self.action, self.gripper = action, gripper
            self.notify(f'{self}; applies when the rollout continues')
        return True
