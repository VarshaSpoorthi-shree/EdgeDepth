"""Headless UI tests with Streamlit's AppTest: every page renders and every main button works.

    python -m pytest -q tests/test_app.py
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["🏠 Home", "🧩 Depth Reconstruction", "🆚 Method Comparison", "📊 Evaluation Dashboard", "🏗️ Technical Architecture",
         "⚠️ Failure Analysis", "🧊 3-D Flying-Pixel Viewer"]


def app():
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=180)
    at.run()
    return at


def goto(at, page):
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    goto(app(), page)


def test_reconstruction_runs_with_metrics():
    at = goto(app(), PAGES[1])
    at.button[0].click().run()
    assert not at.exception and not at.error
    labels = [m.label for m in at.metric]
    assert "Processing time" in labels and any("RMSE" in l for l in labels)


def test_reconstruction_every_method_and_sparse():
    at = goto(app(), PAGES[1])
    at.radio(key="rec_mode").set_value("Sparse samples").run()
    for m in at.selectbox[1].options:
        at.selectbox[1].set_value(m).run()
        at.button[0].click().run()
        assert not at.exception and not at.error, m


def test_comparison_and_dashboard():
    at = goto(app(), PAGES[2])
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert len(at.dataframe) >= 1
    goto(at, PAGES[3])
    assert len(at.dataframe) >= 1


def test_failure_cases_run():
    at = goto(app(), PAGES[5])
    for s in at.selectbox[0].options:
        at.selectbox[0].set_value(s).run()
        at.button[0].click().run()
        assert not at.exception and not at.error, s


def test_reconstruction_shows_confidence_map():
    at = goto(app(), PAGES[1])
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert any("Rank correlation" in m.label for m in at.metric)


def test_viewer3d_builds_all_colour_modes():
    at = goto(app(), PAGES[6])
    for mode in at.selectbox(key="v3d_colour").options:
        at.selectbox(key="v3d_colour").set_value(mode).run()
        at.button[0].click().run()
        assert not at.exception and not at.error, mode
        assert any("Flying pixels" in m.label for m in at.metric)


def test_misalignment_simulation_and_autocorrection():
    at = goto(app(), PAGES[1])
    at.slider(key="rec_shx").set_value(8).run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert any("Misalignment detected" in c.value or "No significant misalignment" in c.value for c in at.caption)
    at.toggle(key="rec_align").set_value(False).run()
    at.button[0].click().run()
    assert not at.exception and not at.error


def test_failure_misalignment_shows_aligned_hybrid():
    at = goto(app(), PAGES[5])
    opt = [o for o in at.selectbox[0].options if "Incorrect boundaries" in o][0]
    at.selectbox[0].set_value(opt).run()
    at.button[0].click().run()
    assert not at.exception and not at.error


def test_reconstruction_warns_on_very_sparse_input():
    at = goto(app(), PAGES[1])
    at.radio(key="rec_mode").set_value("Sparse samples").run()
    at.slider(key="rec_dens").set_value(0.2).run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    texts = [w.value for w in at.warning]
    assert any("Low-reliability" in t for t in texts), texts
