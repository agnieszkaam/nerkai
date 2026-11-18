"""SHAP explanation and waterfall plot (PNG) for one admission.

The figure is a matplotlib Figure that is not registered with pyplot, so Streamlit closing all
pyplot figures after each script run cannot close it while another session is drawing.
"""
import io
import json
import re
import threading
from functools import cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox

from shap_waterfall import waterfall

MODEL_DIR = Path(__file__).resolve().parent / "models"
# matplotlib is not thread-safe and all sessions share one process (st.pyplot docs)
PLOT_LOCK = threading.Lock()


def short_label(label):
    """Label without the ICD-10 codes in brackets."""
    return re.sub(r" \(.*\)$", "", label)


@cache
def load_model(cohort):
    meta = json.loads((MODEL_DIR / f"{cohort}.json").read_text())
    return joblib.load(MODEL_DIR / f"{cohort}.joblib"), meta


@cache
def explainer(cohort):
    """shap.LinearExplainer (log-odds scale) with the training data as background, given as
    the mean and covariance of the transformed features."""
    model, meta = load_model(cohort)
    return shap.LinearExplainer(model.named_steps["model"],
                                (np.array(meta["train_mean"]), np.array(meta["train_cov"])))


def explanation(cohort, row, labels, no_yes):
    model, meta = load_model(cohort)
    X = pd.DataFrame([row])[list(model.feature_names_in_)]
    z = model.named_steps["prep"].transform(X)
    sv = explainer(cohort)(z)[0]
    names = meta["features_out"]
    shown = [str(int(X["wiek"].iloc[0])) if f == "wiek" else no_yes[int(v)]
             for f, v in zip(names, z[0])]
    return shap.Explanation(sv.values, base_values=sv.base_values,
                            data=np.array(shown, dtype=object),
                            feature_names=[short_label(meta[labels][f]) for f in names])


def localise_waterfall(fig, decimal):
    """Decimal separator of the language in shap's waterfall (values, axis, E[f(X)] / f(x))."""
    bars, *value_axes = fig.axes
    bars.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:g}".replace(".", decimal).replace("-", "−")))
    for t in bars.texts:
        t.set_text(t.get_text().replace(".", decimal))
    # shap writes these labels in mathtext, where a bare comma gets punctuation spacing
    math_decimal = "{,}" if decimal == "," else "."
    for ax in value_axes:
        ax.set_xticklabels([t.get_text().replace(".", math_decimal)
                            for t in ax.get_xticklabels()])


def waterfall_png(cohort, row, labels, no_yes, decimal, x_extent, transparent):
    """PNG bytes of the SHAP waterfall with all features, at a fixed width (x_extent, inches)
    and st.pyplot's default resolution (200 dpi)."""
    with PLOT_LOCK:
        sv = explanation(cohort, row, labels, no_yes)
        fig = Figure()
        FigureCanvasAgg(fig)
        waterfall(sv, max_display=len(sv.values), show=False, ax=fig.add_subplot())
        localise_waterfall(fig, decimal)
        box = fig.get_tightbbox()
        png = io.BytesIO()
        fig.savefig(png, format="png", dpi=200, transparent=transparent,
                    bbox_inches=Bbox.from_extents(x_extent[0], box.y0, x_extent[1], box.y1))
    return png.getvalue()
