# Keep OpenCV's native warnings from flooding normal training logs.
# Users can still override this before import, e.g. OPENCV_LOG_LEVEL=WARNING.
import os
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')

# NOTE: the outermost level is hard-coded
__all__ = ['models', 'dataloaders', 'runners']
