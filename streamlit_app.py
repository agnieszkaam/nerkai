"""Calculator of in-hospital mortality risk for AKI / CKD admissions, with a SHAP waterfall
plot per patient. Polish and English.

Run: streamlit run streamlit_app.py

Claude Opus 5.5 (Anthropic), via Claude Code, was used to align the code with library
documentation and best practices. All analytical decisions were made by the author.
"""
import pandas as pd
import streamlit as st

from plotting import load_model, short_label, waterfall_png

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
        "inputs_changed": "Dane zostały zmienione. Kliknij „Oblicz”, aby przeliczyć.",
        "decimal": ",",
        "footnote": "Kalkulator opracowano w ramach abstraktu prezentowanego na IV Forum Młodych "
                    "Naukowców:",
        "abstract_language": "",
        "waterfall_note": "Znaczenie poszczególnych zmiennych w predykcji ryzyka zgonu.",
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
        "inputs_changed": "Inputs have changed. Click “Calculate” to recalculate.",
        "decimal": ".",
        "footnote": "This calculator was developed as part of the abstract presented at the "
                    "IV Forum Młodych Naukowców (4th Young Scientists' Forum):",
        "abstract_language": " [in Polish]",
        "waterfall_note": "Contribution of each variable to the predicted risk of death.",
        "labels": "labels_en",
    },
}


# ~270 KB per plot: at most ~27 MB; repeated inputs are shown without recomputing
@st.cache_data(max_entries=100, show_spinner=False)
def waterfall(cohort, row, language, dark):
    text = TEXT[language]
    # shap text is black: white background on the dark theme
    return waterfall_png(cohort, row, text["labels"], text["no_yes"], text["decimal"], PLOT_X,
                         transparent=not dark)


def percent(x, text):
    return f"{100 * x:.1f}%".replace(".", text["decimal"])


def defaults():
    return {"cohort": COHORTS[0], "wiek": 70, "pilneplanowe": ADMISSION[0], "weekend": 0,
            "result": None} | {c: False for c in sum(chapters.values(), [])}


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

    model, _ = models[cohort]
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

    current = {"cohort": cohort, "row": {c: row[c] for c in model.feature_names_in_}}
    calculate, clear, _ = st.columns([1, 1, 4])
    # the result is computed for the inputs at the last click, not on every input change
    if calculate.button(text["calculate"], type="primary", width="stretch"):
        st.session_state.result = current
    clear.button(text["reset"], on_click=reset, width="stretch")

result = st.session_state.result
shown_cohort = result["cohort"] if result else cohort
shown_model, shown_meta = models[shown_cohort]
reference_risk = text["reference_risk"].format(
    cohort=text["cohorts"][shown_cohort].split(" ")[0], year=shown_meta["reference_risk"]["year"],
    rate=percent(shown_meta["reference_risk"]["mean"], text))
with result_col:
    if result:
        X = pd.DataFrame([result["row"]])[list(shown_model.feature_names_in_)]
        risk = shown_model.predict_proba(X)[0, 1]
        st.markdown(f"**{text['risk']}**")
        st.metric(text["risk"], percent(risk, text), label_visibility="collapsed")
        st.caption(reference_risk)
        if result != current:
            st.caption(f":red[{text['inputs_changed']}]")
        with st.spinner():
            st.image(waterfall(result["cohort"], result["row"], language,
                               st.context.theme.type == "dark"), width="stretch")
        st.caption(text["waterfall_note"])
    else:
        st.markdown(f"**{text['risk']}**")
        st.metric(text["risk"], "—", label_visibility="collapsed")
        st.caption(reference_risk)
        # st.container has no background option: grey fill via its key class
        st.html("<style>.st-key-placeholder {background: rgba(128, 128, 128, 0.1);}</style>")
        with st.container(border=True, height=PLACEHOLDER_HEIGHT, key="placeholder",
                          horizontal_alignment="center", vertical_alignment="center"):
            st.header(":gray[:material/waterfall_chart:]", anchor=False, width="content")
        st.caption(text["waterfall_note"])

st.divider()
st.caption(f"{text['footnote']}  \n{ABSTRACT}{text['abstract_language']}  \n{text['papers']}: "
           + " · ".join(f"[https://doi.org/{doi}](https://doi.org/{doi})" for doi in PAPERS))
