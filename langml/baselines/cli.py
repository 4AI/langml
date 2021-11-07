# -*- coding: utf-8 -*-

import click

from langml.baselines.ner.cli import ner
from langml.baselines.clf.cli import clf


@click.group()
def baseline():
    """LangML Baseline client"""
    pass


baseline.add_command(ner)
baseline.add_command(clf)
