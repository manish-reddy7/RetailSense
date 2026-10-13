"""Smoke test: every dashboard view renders from stored results without raising."""

import pytest
from streamlit.testing.v1 import AppTest

from retailsense.config import ROOT


@pytest.mark.parametrize("view", ["Forecast", "Explain", "Evaluate"])
def test_views_render(view):
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    if at.warning and "No stored runs" in at.warning[0].value:
        pytest.skip("no stored runs in this checkout")
    at.sidebar.radio[0].set_value(view).run()
    assert not at.exception, at.exception
