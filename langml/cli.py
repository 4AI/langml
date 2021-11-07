# -*- coding: utf-8 -*-

import click

import langml


@click.group()
@click.version_option(version=langml.__version__)
def cli():
    """LangML client"""
    pass


def main():
    from langml.baselines.cli import baseline

    cli.add_command(baseline)
    cli(prog_name='langml', obj={})
