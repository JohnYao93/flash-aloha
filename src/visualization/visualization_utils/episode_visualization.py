"""Shared exact-reference raw reader for offline Rerun visualization."""
from data_processing.episode_data import Episode, ReferencedFrames, resolve_episode_directory


ReferencedVideo = ReferencedFrames


class VisualizationEpisode(Episode):
    def __init__(self, directory, *, transform=lambda image: image):
        super().__init__(directory, inspection=True, transform=transform)
