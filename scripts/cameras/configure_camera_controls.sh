#!/usr/bin/env bash
v4l2-ctl -d /dev/video_left --set-ctrl=exposure_dynamic_framerate=0
v4l2-ctl -d /dev/video_right --set-ctrl=exposure_dynamic_framerate=0
v4l2-ctl -d /dev/video_overhead --set-ctrl=exposure_dynamic_framerate=0
# v4l2-ctl -d /dev/video_torso --set-ctrl=exposure_dynamic_framerate=0
# v4l2-ctl -d /dev/video_test --set-ctrl=exposure_dynamic_framerate=0

