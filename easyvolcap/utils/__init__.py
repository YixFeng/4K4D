def configure_opencv_logging(cv2, level: str = 'ERROR'):
    log_level = getattr(cv2, f'LOG_LEVEL_{level}', None)
    if log_level is not None and hasattr(cv2, 'setLogLevel'):
        cv2.setLogLevel(log_level)
        return

    try:
        from cv2.utils import logging
    except ImportError:
        return

    log_level = getattr(logging, f'LOG_LEVEL_{level}', None)
    if log_level is not None:
        logging.setLogLevel(log_level)


# do not try to import all modules (using import_submodules) because of circular import
# utilities should be used as is
