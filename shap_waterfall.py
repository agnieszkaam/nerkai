"""shap.plots.waterfall from shap 0.51.0, changed only to draw on a given matplotlib Axes (ax)
instead of pyplot's current figure: each plt.* call is the same call on ax (plt.barh -> ax.barh,
plt.yticks -> ax.set_yticks, plt.gcf() -> ax.figure, ...). Without ax it behaves as the original.

Streamlit closes all pyplot figures after every script run, which broke the original when
several sessions drew at once (it draws through pyplot's current figure).

Original: https://github.com/shap/shap/blob/v0.51.0/shap/plots/_waterfall.py

The MIT License (MIT)

Copyright (c) 2018 Scott Lundberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shap import Explanation
from shap.plots._labels import labels
from shap.plots._style import get_style
from shap.utils import format_value


def waterfall(shap_values, max_display=10, show=True, ax=None):
    """Plots an explanation of a single prediction as a waterfall plot.

    The SHAP value of a feature represents the impact of the evidence provided by that feature on the model's
    output. The waterfall plot is designed to visually display how the SHAP values (evidence) of each feature
    move the model output from our prior expectation under the background data distribution, to the final model
    prediction given the evidence of all the features.

    Features are sorted by the magnitude of their SHAP values with the smallest
    magnitude features grouped together at the bottom of the plot when the number of
    features in the models exceeds the ``max_display`` parameter.

    Parameters
    ----------
    shap_values : Explanation
        A one-dimensional :class:`.Explanation` object that contains the feature values and SHAP values to plot.

    max_display : int
        The maximum number of features to display (default is 10).

    show : bool
        Whether :external+mpl:func:`matplotlib.pyplot.show()` is called before returning.
        Setting this to ``False`` allows the plot to be customized further after it
        has been created, returning the current axis via plt.gca().

    Examples
    --------
    See `waterfall plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/waterfall.html>`_.

    """
    style = get_style()
    if ax is None:
        ax = plt.gca()
    fig = ax.figure
    # Turn off interactive plot
    if show is False:
        plt.ioff()

    # make sure the input is an Explanation object
    if not isinstance(shap_values, Explanation):
        emsg = "The waterfall plot requires an `Explanation` object as the `shap_values` argument."
        raise TypeError(emsg)

    # make sure we only have a single explanation to plot
    sv_shape = shap_values.shape
    if len(sv_shape) != 1:
        emsg = (
            "The waterfall plot can currently only plot a single explanation, but a "
            f"matrix of explanations (shape {sv_shape}) was passed! Perhaps try "
            "`shap.plots.waterfall(shap_values[0])` or for multi-output models, "
            "try `shap.plots.waterfall(shap_values[0, 0])`."
        )
        raise ValueError(emsg)

    base_values = float(shap_values.base_values)
    features = shap_values.display_data if shap_values.display_data is not None else shap_values.data
    feature_names = shap_values.feature_names
    lower_bounds = getattr(shap_values, "lower_bounds", None)
    upper_bounds = getattr(shap_values, "upper_bounds", None)
    values = shap_values.values

    # unwrap pandas series
    if isinstance(features, pd.Series):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # fallback feature names
    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(len(values))])

    # init variables we use for tracking the plot locations
    num_features = min(max_display, len(values))
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_values + values.sum()
    yticklabels = ["" for _ in range(num_features + 1)]

    # size the plot based on how many features we are plotting
    fig.set_size_inches(8, num_features * row_height + 1.5)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(values):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            ax.plot(
                [loc, loc],
                [rng[i] - 1 - 0.4, rng[i] + 0.4],
                color=style.vlines_color,
                linestyle="--",
                linewidth=0.5,
                zorder=-1,
            )
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            if np.issubdtype(type(features[order[i]]), np.number):
                yticklabels[rng[i]] = (
                    format_value(float(features[order[i]]), "%0.03f") + " = " + str(feature_names[order[i]])
                )
            else:
                yticklabels[rng[i]] = str(features[order[i]]) + " = " + str(feature_names[order[i]])

    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(values):
        yticklabels[0] = f"{len(shap_values) - num_features + 1} other features"
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)

    points = (
        pos_lefts
        + list(np.array(pos_lefts) + np.array(pos_widths))
        + neg_lefts
        + list(np.array(neg_lefts) + np.array(neg_widths))
    )
    dataw = np.max(points) - np.min(points)

    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1 * dataw if w < 1 else 0 for w in pos_widths])
    ax.barh(
        pos_inds,
        np.array(pos_widths) + label_padding + 0.02 * dataw,
        left=np.array(pos_lefts) - 0.01 * dataw,
        color=style.primary_color_positive,
        alpha=0,
    )
    label_padding = np.array([-0.1 * dataw if -w < 1 else 0 for w in neg_widths])
    ax.barh(
        neg_inds,
        np.array(neg_widths) + label_padding - 0.02 * dataw,
        left=np.array(neg_lefts) + 0.01 * dataw,
        color=style.primary_color_negative,
        alpha=0,
    )

    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = ax.get_xlim()[1] - ax.get_xlim()[0]
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen / width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()

    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = ax.arrow(
            pos_lefts[i],
            pos_inds[i],
            dist - hl_scaled,
            0,
            head_length=min(dist, hl_scaled),
            color=style.primary_color_positive,
            width=bar_width,
            head_width=bar_width,
        )

        if pos_low is not None and i < len(pos_low):
            ax.errorbar(
                pos_lefts[i] + pos_widths[i],
                pos_inds[i],
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor=style.secondary_color_positive,
            )

        txt_obj = ax.text(
            pos_lefts[i] + 0.5 * dist,
            pos_inds[i],
            format_value(pos_widths[i], "%+0.02f"),
            horizontalalignment="center",
            verticalalignment="center",
            color=style.text_color,
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = ax.text(
                pos_lefts[i] + (5 / 72) * bbox_to_xscale + dist,
                pos_inds[i],
                format_value(pos_widths[i], "%+0.02f"),
                horizontalalignment="left",
                verticalalignment="center",
                color=style.primary_color_positive,
                fontsize=12,
            )

    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]

        arrow_obj = ax.arrow(
            neg_lefts[i],
            neg_inds[i],
            -(-dist - hl_scaled),
            0,
            head_length=min(-dist, hl_scaled),
            color=style.primary_color_negative,
            width=bar_width,
            head_width=bar_width,
        )

        if neg_low is not None and i < len(neg_low):
            ax.errorbar(
                neg_lefts[i] + neg_widths[i],
                neg_inds[i],
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor=style.secondary_color_negative,
            )

        txt_obj = ax.text(
            neg_lefts[i] + 0.5 * dist,
            neg_inds[i],
            format_value(neg_widths[i], "%+0.02f"),
            horizontalalignment="center",
            verticalalignment="center",
            color=style.text_color,
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = ax.text(
                neg_lefts[i] - (5 / 72) * bbox_to_xscale + dist,
                neg_inds[i],
                format_value(neg_widths[i], "%+0.02f"),
                horizontalalignment="right",
                verticalalignment="center",
                color=style.primary_color_negative,
                fontsize=12,
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ytick_pos = list(range(num_features)) + list(np.arange(num_features) + 1e-8)
    ax.set_yticks(ytick_pos, yticklabels[:-1] + [label.split("=")[-1] for label in yticklabels[:-1]], fontsize=13)

    # put horizontal lines for each feature row
    for i in range(num_features):
        ax.axhline(i, color=style.hlines_color, lw=0.5, dashes=(1, 5), zorder=-1)

    # mark the prior expected value and the model prediction
    ax.axvline(base_values, 0, 1 / num_features, color=style.vlines_color, linestyle="--", linewidth=0.5, zorder=-1)
    fx = base_values + values.sum()
    ax.axvline(fx, 0, 1, color=style.vlines_color, linestyle="--", linewidth=0.5, zorder=-1)

    # clean up the main axis
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=13)
    # plt.xlabel("\nModel output", fontsize=12)

    # draw the E[f(X)] tick mark
    xmin, xmax = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks(
        [base_values, base_values + min(1e-8, xmax * 1e-10)]
    )  # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    # However, for very small values, 1e-8 is disruptively large, so xmax * 1e-10 is used instead
    ax2.set_xticklabels(["\n$E[f(X)]$", "\n$ = " + format_value(base_values, "%0.03f") + "$"], fontsize=12, ha="left")
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # draw the f(x) tick mark
    ax3 = ax2.twiny()
    ax3.set_xlim(xmin, xmax)
    ax3.set_xticks(
        [base_values + values.sum(), base_values + values.sum() + min(1e-8, xmax * 1e-10)]
    )  # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    # However, for very small values, 1e-8 is disruptively large, so xmax * 1e-10 is used instead
    ax3.set_xticklabels(["$f(x)$", "$ = " + format_value(fx, "%0.03f") + "$"], fontsize=12, ha="left")
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-10 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(12 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_color(style.tick_labels_color)
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["left"].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-20 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform()
        + matplotlib.transforms.ScaledTranslation(22 / 72.0, -1 / 72.0, fig.dpi_scale_trans)
    )

    tick_labels[1].set_color(style.tick_labels_color)

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color(style.tick_labels_color)

    if show:
        plt.show()
    else:
        return ax

