#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `real_robots` package."""

import pytest

from click.testing import CliRunner

from real_robots import real_robots  # noqa
from real_robots import cli


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.demo)
    assert result.exit_code == 0
    help_result = runner.invoke(cli.demo, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output
