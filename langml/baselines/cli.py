# -*- coding: utf-8 -*-

import click

from langml.baselines.ner.cli import ner
from langml.baselines.clf.cli import clf
from langml.baselines.contrastive.cli import contrastive


@click.group()
def baseline():
    """LangML Baseline client"""
    pass


baseline.add_command(ner)
baseline.add_command(clf)
baseline.add_command(contrastive)
