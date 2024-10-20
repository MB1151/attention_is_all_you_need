import logging

def get_logger(name, level=None):
    """Sets up a logger with the given name and level.

    Args:
        name (_type_): Name of the logger. Usually the name of the module.
        level (_type_, optional): Level of the logger. Defaults to logging.DEBUG.

    Returns:
        _type_: Returns the logger object.
    """
    # This is pretty useful to understand the flow of the program and to debug issues.
    formatter = logging.Formatter('[%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s]')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    # If the level is not provided, then it should be inhertied from the parent logger.
    if level is not None:
        logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger