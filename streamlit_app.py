"""Calculator of in-hospital mortality risk for AKI / CKD admissions, with a SHAP waterfall
plot per patient. Polish and English.

Run: streamlit run streamlit_app.py
"""
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from matplotlib.transforms import Bbox

MODEL_DIR = Path(__file__).resolve().parent / "models"
COHORTS = ["aki", "ckd"]
ADMISSION = ["scheduled", "emergency"]
CHAPTERS = {
    "C": ["solid_tumor", "metastatic_cancer", "lymphoma"],
    "D": ["coagulopathy", "blood_loss_anemia", "deficiency_anemia"],
    "E": ["diabetes_uncomplicated", "diabetes_complicated", "hypothyroidism", "obesity",
          "weight_loss"],
    "I": ["chf", "arrhythmia", "valvular_disease", "pulmonary_circ_disorders", "pvd",
          "ht_uncomplicated", "ht_complicated"],
    "J": ["chronic_pulm_disease"],
    "K": ["liver_disease", "peptic_ulcer"],
}
LEFT_CHAPTERS = ["C", "D", "E"]
ABSTRACT = ("Mazurek A, Grudziąż-Sękowska J, Jankowski M. Modele uczenia maszynowego w "
            "przewidywaniu zgonu wewnątrzszpitalnego i przedłużonej hospitalizacji u pacjentów z "
            "ostrym uszkodzeniem nerek i przewlekłą chorobą nerek.")
PAPERS = ["10.5114/aoms/210331", "10.1186/s12882-026-04885-2"]
# fixed plot width (inches): same image size in both languages and for every patient
PLOT_X = (-2.9, 7.9)
PLACEHOLDER_HEIGHT = 420  # px
TEXT = {
    "PL": {
        "page_title": "Kalkulator ryzyka zgonu",
        "title": "Kalkulator ryzyka zgonu wewnątrzszpitalnego",
        "caption": "Prototyp. Niezwalidowany klinicznie, wyłącznie do celów badawczych.",
        "papers": "Powiązane publikacje",
        "diagnosis": "Rozpoznanie zasadnicze",
        "cohorts": {"aki": "AKI (N17)", "ckd": "PChN (N18)"},
        "age": "Wiek (w latach)",
        "admission": "Tryb przyjęcia",
        "admission_types": {"scheduled": "planowy", "emergency": "nagły"},
        "no_yes": ["nie", "tak"],
        "comorbidities": "Choroby współistniejące (wg rozdziałów ICD-10)",
        "present_on_admission": "Obecne w chwili przyjęcia do szpitala.",
        "chapters": {"C": "Nowotwory (C)", "D": "Choroby krwi i narządów krwiotwórczych (D)",
                     "E": "Choroby endokrynologiczne i metaboliczne (E)",
                     "I": "Choroby układu krążenia (I)", "J": "Choroby układu oddechowego (J)",
                     "K": "Choroby układu trawiennego (K)"},
        "not_in_model": "Nieuwzględnione w modelu PChN.",
        "calculate": "Oblicz",
        "reset": "Resetuj",
        "risk": "Przewidywane ryzyko zgonu wewnątrzszpitalnego",
        "reference_risk": "Średnie przewidywane ryzyko w {cohort} ({year}): {rate}",
        "decimal": ",",
        "footnote": "Kalkulator opracowano w ramach abstraktu prezentowanego na IV Forum Młodych "
                    "Naukowców:",
        "abstract_language": "",
        "labels": "labels",
    },
    "EN": {
        "page_title": "Mortality risk calculator",
        "title": "In-hospital mortality risk calculator",
        "caption": "Prototype. Not clinically validated, for research use only.",
        "papers": "Related publications",
        "diagnosis": "Principal diagnosis",
        "cohorts": {"aki": "AKI (N17)", "ckd": "CKD (N18)"},
        "age": "Age (years)",
        "admission": "Admission type",
        "admission_types": {"scheduled": "elective", "emergency": "emergency"},
        "no_yes": ["no", "yes"],
        "comorbidities": "Comorbidities (by ICD-10 chapter)",
        "present_on_admission": "Present on admission.",
        "chapters": {"C": "Neoplasms (C)", "D": "Diseases of the blood (D)",
                     "E": "Endocrine, nutritional and metabolic diseases (E)",
                     "I": "Diseases of the circulatory system (I)",
                     "J": "Diseases of the respiratory system (J)",
                     "K": "Diseases of the digestive system (K)"},
        "not_in_model": "Not included in the CKD model.",
        "calculate": "Calculate",
        "reset": "Reset",
        "risk": "Predicted in-hospital mortality risk",
        "reference_risk": "Average predicted risk in {cohort} ({year}): {rate}",
        "decimal": ".",
        "footnote": "This calculator was developed as part of the abstract presented at the "
                    "IV Forum Młodych Naukowców (4th Young Scientists' Forum):",
        "abstract_language": " [in Polish]",
        "labels": "labels_en",
    },
}


@st.cache_resource
def load_model(cohort):
    meta = json.loads((MODEL_DIR / f"{cohort}.json").read_text())
    return joblib.load(MODEL_DIR / f"{cohort}.joblib"), meta


def percent(x, text):
    return f"{100 * x:.1f}%".replace(".", text["decimal"])


def short_label(label):
    """Label without the ICD-10 codes in brackets."""
    return re.sub(r" \(.*\)$", "", label)


def explanation(model, meta, X, text):
    """SHAP values of the logistic regression with the training data as background:
    coef * (x - training mean) on the log-odds scale."""
    prep, lr = model.named_steps["prep"], model.named_steps["model"]
    z = prep.transform(X)[0]
    names = meta["features_out"]
    shown = [str(int(X["wiek"].iloc[0])) if f == "wiek" else text["no_yes"][int(v)]
             for f, v in zip(names, z)]
    return shap.Explanation(lr.coef_[0] * (z - np.array(meta["train_mean"])),
                            base_values=meta["reference_logodds"],
                            data=np.array(shown, dtype=object),
                            feature_names=[short_label(meta[text["labels"]][f]) for f in names])


def defaults():
    return {"cohort": COHORTS[0], "wiek": 70, "pilneplanowe": ADMISSION[0], "weekend": 0,
            "calculated": False} | {c: False for c in sum(chapters.values(), [])}


def reset():
    st.session_state.update(defaults())


models = {cohort: load_model(cohort) for cohort in COHORTS}
codes = {k: v for _, meta in models.values() for k, v in meta["icd10_codes"].items()}
used = {c for model, _ in models.values() for c in model.feature_names_in_}
chapters = {ch: [c for c in cols if c in used] for ch, cols in CHAPTERS.items()}
chapters = {ch: cols for ch, cols in chapters.items() if cols}
assert used <= {"wiek", "pilneplanowe", "weekend", *sum(chapters.values(), [])}, \
    "model input not in the form"
for key, value in defaults().items():
    st.session_state.setdefault(key, value)
    # keeps inputs when their labels change language (Streamlit widget-state docs)
    st.session_state[key] = st.session_state[key]

language = st.session_state.get("language", "PL")
text = TEXT[language]
labels = {k: v for _, meta in models.values() for k, v in meta[text["labels"]].items()}

st.set_page_config(page_title=text["page_title"], layout="wide")
title, switch = st.columns([6, 1], vertical_alignment="center")
title.header(text["title"])
switch.radio("PL/EN", list(TEXT), horizontal=True, key="language", label_visibility="collapsed")
st.caption(text["caption"])
inputs_col, result_col = st.columns(2, gap="large")

with inputs_col:
    diagnosis, age = st.columns(2)
    cohort = diagnosis.radio(text["diagnosis"], COHORTS, format_func=text["cohorts"].get,
                             horizontal=True, key="cohort")
    row = {"wiek": age.number_input(text["age"], min_value=18, max_value=110, key="wiek")}
    admission, weekend = st.columns(2)
    row["pilneplanowe"] = admission.radio(text["admission"], ADMISSION,
                                          format_func=text["admission_types"].get,
                                          horizontal=True, key="pilneplanowe")
    row["weekend"] = weekend.radio(labels["weekend"], [0, 1],
                                   format_func=lambda v: text["no_yes"][v], horizontal=True,
                                   key="weekend")

    model, meta = models[cohort]
    in_model = set(model.feature_names_in_)
    for c in sum(chapters.values(), []):
        if c not in in_model:
            st.session_state[c] = False
    st.markdown(f"**{text['comorbidities']}** :small[:gray[{text['present_on_admission']}]]")
    left, right = st.columns(2)
    for chapter, cols in chapters.items():
        with left if chapter in LEFT_CHAPTERS else right:
            st.badge(text["chapters"][chapter], color="blue")
            for c in cols:
                row[c] = int(st.checkbox(short_label(labels[c]), key=c,
                                         disabled=c not in in_model,
                                         help=", ".join(codes[c]) if c in in_model else
                                         f"{text['not_in_model']}  \n" + ", ".join(codes[c])))

    calculate, clear, _ = st.columns([1, 1, 4])
    if calculate.button(text["calculate"], type="primary", width="stretch"):
        st.session_state.calculated = True
    clear.button(text["reset"], on_click=reset, width="stretch")

reference_risk = text["reference_risk"].format(cohort=text["cohorts"][cohort].split(" ")[0],
                                               year=meta["reference_risk"]["year"],
                                               rate=percent(meta["reference_risk"]["mean"], text))
with result_col:
    if st.session_state.calculated:
        X = pd.DataFrame([row])[list(model.feature_names_in_)]
        risk = model.predict_proba(X)[0, 1]
        st.markdown(f"**{text['risk']}**")
        st.metric(text["risk"], percent(risk, text), label_visibility="collapsed")
        st.caption(reference_risk)
        with st.spinner():
            sv = explanation(model, meta, X, text)
            shap.plots.waterfall(sv, max_display=len(sv.values), show=False)
            fig = plt.gcf()
            box = fig.get_tightbbox()
            # shap text is black: white background on the dark theme
            st.pyplot(fig, clear_figure=True, transparent=st.context.theme.type != "dark",
                      bbox_inches=Bbox.from_extents(PLOT_X[0], box.y0, PLOT_X[1], box.y1))
    else:
        st.markdown(f"**{text['risk']}**")
        st.metric(text["risk"], "—", label_visibility="collapsed")
        st.caption(reference_risk)
        # st.container has no background option: grey fill via its key class
        st.html("<style>.st-key-placeholder {background: rgba(128, 128, 128, 0.1);}</style>")
        with st.container(border=True, height=PLACEHOLDER_HEIGHT, key="placeholder",
                          horizontal_alignment="center", vertical_alignment="center"):
            st.header(":gray[:material/waterfall_chart:]", anchor=False, width="content")

st.divider()
st.caption(f"{text['footnote']}  \n{ABSTRACT}{text['abstract_language']}  \n{text['papers']}: "
           + " · ".join(f"[https://doi.org/{doi}](https://doi.org/{doi})" for doi in PAPERS))
