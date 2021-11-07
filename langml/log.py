# -*- coding: utf-8 -*-

import logging
from datetime import datetime
from functools import partial

logging.addLevelName(logging.WARN, 'WARN')


def print_log(level: int, msg: str, *args):
    log = '%s %s %s' % (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        logging.getLevelName(level),
        msg % args
    )
    print(log)


debug = partial(print_log, logging.DEBUG)
info = partial(print_log, logging.INFO)
warn = partial(print_log, logging.WARN)
error = partial(print_log, logging.ERROR)
