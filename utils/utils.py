import os
import logging
import logging.config
import platform

__ALL__ = ["LOGGER", "colorstr", 'emojis']

LOGGING_NAME = 'Multitasking-Model'

def set_logging(name=LOGGING_NAME, verbose=True):
    ''' Logging configs, based on YOLOv5 repository.
    '''
    rank = int(os.getenv('RANK', -1))
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            name: {
                "format": "%(message)s"
            }
        },
        "handlers": {
            name: {
                "class": "logging.StreamHandler",
                "formatter": name,
                "level": level
            }
        },
        "loggers": {
            name: {
                "level": level,
                "handlers": [name],
                "propagate": False
            }
        }
    })

set_logging(LOGGING_NAME)

LOGGER = logging.getLogger(LOGGING_NAME)

def colorstr(*input) -> str:
    """ String colorization and style

    Args:
        *inputs: these inputs are `color`, `style`, `str` respectively.
            (default: `('blue', 'bold', str)`)

    Return:
        str: Colored string
    """
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str

