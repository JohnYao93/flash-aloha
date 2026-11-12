"""Two-pedal foot switch keys shared by the collectors and review replay.

The left pedal types '1': arm/start recording, stop recording, accept a take,
start a replay and stop a replay. The right pedal types 'h': reject a take or
cancel a replay. 'h' is also the damping home key (hardware.safe_exit and the
replay stop), so in damping the right pedal returns the arm home.
"""

GO_KEY = '1'
REJECT_KEY = 'h'
REJECT_KEYS = (REJECT_KEY, 'n')
# Moved off '1' so the go pedal stops recording instead of stamping a failure.
FAILURE_STAMP_KEY = 'f'
