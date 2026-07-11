"""Live demo — Subgroup-Disaggregated Faithfulness for SHAP & LIME.

Run:  streamlit run demo/app.py
Requires demo/artifacts/ produced by:  python demo/precompute.py
"""
import os
import pickle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ART = os.path.join(ROOT, "demo", "artifacts")

# ---- palette (validated reference palette, light mode) ----
C_SHAP = "#2a78d6"      # categorical slot 1 (blue)
C_LIME = "#1baf7a"      # categorical slot 2 (aqua)
C_POS = "#2a78d6"       # diverging pair: blue ...
C_NEG = "#e34948"       # ... red
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_CRITICAL = "#d03b3b"

ATTRIBUTES = {
    "sex": ("sex_raw", ["Male", "Female"]),
    "race": ("race_raw", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]),
    "age_group": ("age_group", ["<25", "25-35", "35-50", "50-65", "65+"]),
}
FD_THRESHOLD = 0.1

st.set_page_config(page_title="Subgroup-Disaggregated Faithfulness", page_icon="🔍", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading model ...")
def load_model():
    model = pickle.load(open(os.path.join(DATA, "model_rf.pkl"), "rb"))
    feature_names = pickle.load(open(os.path.join(DATA, "feature_names.pkl"), "rb"))
    return model, feature_names


@st.cache_data(show_spinner="Loading data ...")
def load_data():
    test = pd.read_csv(os.path.join(DATA, "adult_test.csv"))
    idx = np.load(os.path.join(ART, "sample_indices.npy"))
    shap_vals = np.load(os.path.join(ART, "shap_values.npy"))
    lime_path = os.path.join(ART, "lime_values.npy")
    lime_vals = np.load(lime_path) if os.path.exists(lime_path) else None
    sample = test.loc[idx]
    return test, sample, shap_vals, lime_vals


@st.cache_resource(show_spinner=False)
def shap_explainer():
    import shap

    model, _ = load_model()
    return shap.TreeExplainer(model)


@st.cache_resource(show_spinner="Preparing LIME explainer ...")
def lime_explainer():
    from lime.lime_tabular import LimeTabularExplainer

    _, feature_names = load_model()
    train = pd.read_csv(os.path.join(DATA, "adult_train.csv"))
    return LimeTabularExplainer(
        train[feature_names].values,
        feature_names=feature_names,
        class_names=["<=50K", ">50K"],
        mode="classification",
        random_state=42,
    )


# ---------------------------------------------------------------- metrics
def _proba(M, model, feature_names):
    return model.predict_proba(pd.DataFrame(M, columns=feature_names))[:, 1]


@st.cache_data(show_spinner="Computing PGI / PGU ...")
def compute_pgi_pgu(k: int):
    """Per-instance PGI/PGU for both explainers — same code as notebook 03."""
    model, feature_names = load_model()
    _, sample, shap_vals, lime_vals = load_data()
    X = sample[feature_names].values.astype(float)
    means = X.mean(axis=0)
    fx = _proba(X, model, feature_names)

    out = {}
    for name, phi in (("SHAP", shap_vals), ("LIME", lime_vals)):
        if phi is None:
            continue
        order = np.argsort(np.abs(phi), axis=1)
        top_idx, bot_idx = order[:, -k:], order[:, :k]
        rows = np.repeat(np.arange(len(X)), k)
        X_top, X_bot = X.copy(), X.copy()
        X_top[rows, top_idx.ravel()] = means[top_idx.ravel()]
        X_bot[rows, bot_idx.ravel()] = means[bot_idx.ravel()]
        out[name] = {
            "PGI": fx - _proba(X_top, model, feature_names),
            "PGU": fx - _proba(X_bot, model, feature_names),
        }
    return out, fx


@st.cache_data(show_spinner=False)
def subgroup_table(k: int):
    """Long dataframe: attribute, group, metric, explainer, score, n."""
    _, sample, _, _ = load_data()
    per_inst, _ = compute_pgi_pgu(k)
    rows = []
    for attr, (col, order) in ATTRIBUTES.items():
        for expl, metrics in per_inst.items():
            for metric, vals in metrics.items():
                s = pd.Series(vals, index=sample.index)
                rows.append(
                    {"attribute": attr, "group": "All (global)", "metric": metric,
                     "explainer": expl, "score": float(s.mean()), "n": len(s)}
                )
                for g in order:
                    mask = sample[col] == g
                    if mask.sum() == 0:
                        continue
                    rows.append(
                        {"attribute": attr, "group": g, "metric": metric,
                         "explainer": expl, "score": float(s[mask.values].mean()),
                         "n": int(mask.sum())}
                    )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def disparity_table(k: int):
    df = subgroup_table(k)
    rows = []
    for attr in ATTRIBUTES:
        for metric in ["PGI", "PGU"]:
            for expl in sorted(df["explainer"].unique()):
                sub = df[(df.attribute == attr) & (df.metric == metric)
                         & (df.explainer == expl) & (df.group != "All (global)")]
                scores = sub.set_index("group")["score"]
                gscore = df[(df.attribute == attr) & (df.metric == metric)
                            & (df.explainer == expl) & (df.group == "All (global)")]["score"].iloc[0]
                edi_per_group = gscore - scores
                rows.append({
                    "attribute": attr, "metric": metric, "explainer": expl,
                    "WSF": scores.min(), "WSF_group": scores.idxmin(),
                    "FD": scores.max() - scores.min(),
                    "FD_flag": bool(scores.max() - scores.min() > FD_THRESHOLD),
                    "EDI": edi_per_group.max(), "EDI_group": edi_per_group.idxmax(),
                    "global": gscore,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- charts
def style(chart):
    return (
        chart.configure_view(strokeOpacity=0)
        .configure_axis(gridColor=C_GRID, domainColor="#c3c2b7", labelColor="#52514e",
                        titleColor=C_MUTED, labelFontSize=12, titleFontSize=12)
        .configure_legend(labelColor="#52514e", titleColor=C_MUTED)
    )


def attribution_chart(phi, feature_names, title):
    df = pd.DataFrame({"feature": feature_names, "value": phi})
    df["absv"] = df["value"].abs()
    df = df.sort_values("absv", ascending=False)
    df["sign"] = np.where(df["value"] >= 0, "pushes toward >50K", "pushes toward <=50K")
    base = alt.Chart(df).encode(
        y=alt.Y("feature:N", sort=df["feature"].tolist(), title=None),
        x=alt.X("value:Q", title="attribution (toward >50K →)"),
        color=alt.Color("sign:N",
                        scale=alt.Scale(domain=["pushes toward >50K", "pushes toward <=50K"],
                                        range=[C_POS, C_NEG]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("value:Q", format=".4f")],
    )
    return style(base.mark_bar(size=14, cornerRadiusEnd=4).properties(height=330, title=title))


# ---------------------------------------------------------------- app
model, feature_names = load_model()

missing = not os.path.exists(os.path.join(ART, "shap_values.npy"))
if missing:
    st.title("Subgroup-Disaggregated Faithfulness — live demo")
    st.error(
        "Precomputed artifacts not found. Run this once first (takes a few minutes):\n\n"
        "```\npython demo/precompute.py\n```"
    )
    st.stop()

test, sample, shap_vals, lime_vals = load_data()
X_sample = sample[feature_names]
feature_means = X_sample.values.astype(float).mean(axis=0)

st.title("🔍 Do explanations serve everyone equally?")
st.markdown(
    "**Subgroup-disaggregated faithfulness for SHAP & LIME** — a Random Forest predicts income "
    "on the UCI Adult dataset (accuracy 0.852). We measure how *faithful* its explanations are — "
    "and then break that score down by **sex, race and age** to reveal what the global average hides."
)

with st.sidebar:
    st.header("Settings")
    k = st.slider("k — features masked (top-k / bottom-k)", 1, 13, 5,
                  help="PGI masks the k most important features; PGU masks the k least important.")
    st.markdown("---")
    st.markdown(
        f"**Evaluation sample**: {len(sample)} test instances, stratified by income class.\n\n"
        f"**Explainers**: SHAP (TreeExplainer, exact){' · LIME (tabular)' if lime_vals is not None else ''}"
    )
    if lime_vals is None:
        st.warning("LIME values not precomputed yet — showing SHAP only. "
                   "Finish `python demo/precompute.py` to enable LIME.")
    st.markdown("---")
    st.caption("PGI = prediction drop when masking *important* features (higher = faithful). "
               "PGU = drop when masking *unimportant* features (lower = faithful). "
               "Masked features are replaced by their sample mean.")

tab1, tab2, tab3 = st.tabs(
    ["① Explain one person", "② The aggregation problem", "③ WSF · FD · EDI — the new metrics"]
)

# ================================================================ TAB 1
with tab1:
    st.subheader("A live faithfulness test on a single person")
    st.markdown(
        "Pick a person from the test sample. SHAP explains the model's prediction **live**; "
        "then we *test* the explanation: masking the features it calls important should crash "
        "the prediction (PGI), masking the ones it calls unimportant should barely move it (PGU)."
    )

    per_inst_all, fx_all = compute_pgi_pgu(k)

    labels = [
        f"#{i}  ·  {r.sex_raw}, {r.race_raw}, age {int(r.age)}  ·  p(>50K) = {fx_all[j]:.2f}"
        for j, (i, r) in enumerate(sample.iterrows())
    ]
    if "pick" not in st.session_state:
        st.session_state.pick = 0
    csel, cbtn = st.columns([4, 1])
    with cbtn:
        st.write("")
        if st.button("🎲 Surprise me"):
            st.session_state.pick = int(np.random.randint(len(labels)))
    with csel:
        pos = st.selectbox("Person", range(len(labels)), format_func=lambda i: labels[i],
                           index=st.session_state.pick, key="person_select")
    row = sample.iloc[pos]
    x = X_sample.iloc[[pos]].values.astype(float)
    p0 = float(fx_all[pos])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sex", row.sex_raw)
    c2.metric("Race", row.race_raw)
    c3.metric("Age", int(row.age))
    c4.metric("Hours / week", int(row["hours-per-week"]))
    c5.metric("True income", ">50K" if row.income == 1 else "≤50K")

    st.markdown("---")
    use_lime_live = st.toggle("Also compute a LIME explanation live (~5 s)", value=False)

    colA, colB = st.columns(2)
    with colA:
        phi_shap = shap_vals[pos]
        st.altair_chart(attribution_chart(phi_shap, feature_names, "SHAP attributions"),
                        width="stretch")
    with colB:
        if use_lime_live:
            with st.spinner("Running LIME (1,000 perturbations) ..."):
                exp = lime_explainer().explain_instance(
                    x[0], model.predict_proba, num_features=len(feature_names), num_samples=1000
                )
            phi_l = np.zeros(len(feature_names))
            for fi, w in exp.local_exp[1]:
                phi_l[fi] = w
            st.altair_chart(attribution_chart(phi_l, feature_names, "LIME attributions (live)"),
                            width="stretch")
        elif lime_vals is not None:
            st.altair_chart(
                attribution_chart(lime_vals[pos], feature_names, "LIME attributions (precomputed)"),
                width="stretch")
        else:
            st.info("Toggle above to run LIME live for this person.")

    # ---- the masking experiment ----
    st.markdown(f"#### The faithfulness test (k = {k})")
    order = np.argsort(np.abs(phi_shap))
    top_idx, bot_idx = order[-k:], order[:k]
    x_top, x_bot = x.copy(), x.copy()
    x_top[0, top_idx] = feature_means[top_idx]
    x_bot[0, bot_idx] = feature_means[bot_idx]
    p_top = float(_proba(x_top, model, feature_names)[0])
    p_bot = float(_proba(x_bot, model, feature_names)[0])

    m1, m2, m3 = st.columns(3)
    m1.metric("Original prediction p(>50K)", f"{p0:.3f}")
    m2.metric(f"Mask top-{k} important features", f"{p_top:.3f}", delta=f"{p_top - p0:+.3f}",
              delta_color="inverse", help="Large move = SHAP found the real drivers (PGI)")
    m3.metric(f"Mask bottom-{k} unimportant features", f"{p_bot:.3f}", delta=f"{p_bot - p0:+.3f}",
              delta_color="off", help="Small move = SHAP correctly ignored these (PGU)")
    st.caption(
        f"Masked important: {', '.join(feature_names[i] for i in reversed(top_idx))} — "
        f"masked unimportant: {', '.join(feature_names[i] for i in bot_idx)}. "
        f"PGI = {p0 - p_top:+.3f}, PGU = {p0 - p_bot:+.3f} for this person (SHAP)."
    )

# ================================================================ TAB 2
with tab2:
    st.subheader("The global average looks fine — until you disaggregate it")
    df = subgroup_table(k)
    expls = sorted(df["explainer"].unique())
    metric = st.radio("Faithfulness metric", ["PGI", "PGU"], horizontal=True,
                      captions=["higher = more faithful", "lower = more faithful"])
    attr = st.radio("Break down by", list(ATTRIBUTES), horizontal=True, key="attr2")

    col_order = ["All (global)"] + ATTRIBUTES[attr][1]
    sub = df[(df.attribute == attr) & (df.metric == metric)].copy()
    sub["is_global"] = sub["group"] == "All (global)"

    gvals = sub[sub.is_global].set_index("explainer")["score"]
    bars = (
        alt.Chart(sub[~sub.is_global])
        .mark_bar(size=18, cornerRadiusEnd=4)
        .encode(
            x=alt.X("group:N", sort=col_order[1:], title=None,
                    axis=alt.Axis(labelAngle=-20)),
            xOffset=alt.XOffset("explainer:N"),
            y=alt.Y("score:Q", title=f"mean {metric}"),
            color=alt.Color("explainer:N",
                            scale=alt.Scale(domain=expls,
                                            range=[C_LIME, C_SHAP] if expls[0] == "LIME" else [C_SHAP, C_LIME]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["group", "explainer", alt.Tooltip("score:Q", format=".4f"),
                     alt.Tooltip("n:Q", title="sample size")],
        )
    )
    rules = alt.Chart(pd.DataFrame({"explainer": gvals.index, "score": gvals.values})).mark_rule(
        strokeDash=[5, 4], size=2
    ).encode(y="score:Q",
             color=alt.Color("explainer:N", legend=None,
                             scale=alt.Scale(domain=expls,
                                             range=[C_LIME, C_SHAP] if expls[0] == "LIME" else [C_SHAP, C_LIME])),
             tooltip=[alt.Tooltip("explainer"), alt.Tooltip("score:Q", format=".4f", title="global mean")])
    st.altair_chart(style((bars + rules).properties(height=380)), width="stretch")
    st.caption("Dashed lines = the **global average** each explainer would report. "
               "Bars = what each subgroup actually gets. Hover for exact values and group sizes.")

    small = sub[(~sub.is_global) & (sub.n < 15)]
    if len(small):
        st.warning("Small subgroups in the 500-instance sample — interpret with care: "
                   + ", ".join(f"{g} (n={n})" for g, n in
                               small[["group", "n"]].drop_duplicates().itertuples(index=False)))

    with st.expander("Show the numbers"):
        piv = sub.pivot_table(index="group", columns="explainer", values="score").reindex(col_order)
        ns = sub.drop_duplicates("group").set_index("group")["n"].reindex(col_order)
        piv["n"] = ns
        st.dataframe(piv.style.format({e: "{:.4f}" for e in expls}))

# ================================================================ TAB 3
with tab3:
    st.subheader("Three new metrics that make the disparity visible")
    st.markdown(
        "- **WSF — Worst Subgroup Faithfulness**: the score of the *worst-served* group. \n"
        "- **FD — Faithfulness Disparity**: best group − worst group. FD > "
        f"{FD_THRESHOLD} is flagged as concerning. \n"
        "- **EDI — Explanation Desert Index**: how far the most abandoned group falls "
        "below the global average."
    )
    disp = disparity_table(k)
    expls = sorted(disp["explainer"].unique())

    metric3 = st.radio("Metric", ["PGI", "PGU"], horizontal=True, key="metric3")
    d = disp[disp.metric == metric3].copy()

    fd_chart = (
        alt.Chart(d)
        .mark_bar(size=18, cornerRadiusEnd=4)
        .encode(
            x=alt.X("attribute:N", title=None, sort=list(ATTRIBUTES)),
            xOffset="explainer:N",
            y=alt.Y("FD:Q", title="Faithfulness Disparity (FD)"),
            color=alt.Color("explainer:N",
                            scale=alt.Scale(domain=expls,
                                            range=[C_LIME, C_SHAP] if expls[0] == "LIME" else [C_SHAP, C_LIME]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["attribute", "explainer", alt.Tooltip("FD:Q", format=".4f"),
                     alt.Tooltip("WSF_group:N", title="worst group")],
        )
    )
    threshold = alt.Chart(pd.DataFrame({"y": [FD_THRESHOLD]})).mark_rule(
        color=C_CRITICAL, strokeDash=[5, 4], size=2
    ).encode(y="y:Q", tooltip=alt.value(f"FD concern threshold = {FD_THRESHOLD}"))
    st.altair_chart(style((fd_chart + threshold).properties(height=340)), width="stretch")
    st.caption(f"Red dashed line: FD = {FD_THRESHOLD} concern threshold from the paper.")

    st.markdown("#### Who is being left behind?")
    cols = st.columns(len(ATTRIBUTES))
    for col, attr in zip(cols, ATTRIBUTES):
        with col:
            for expl in expls:
                r = d[(d.attribute == attr) & (d.explainer == expl)].iloc[0]
                flag = "🚨 " if r.FD_flag else ""
                st.metric(f"{expl} · {attr}", r.EDI_group,
                          delta=f"EDI {r.EDI:+.3f}", delta_color="inverse",
                          help=f"{flag}Group furthest below the global {metric3} average. "
                               f"FD = {r.FD:.4f} (worst group by WSF: {r.WSF_group})")

    flagged = disp[disp.FD_flag]
    if len(flagged):
        st.error(
            f"**{len(flagged)} of {len(disp)}** (metric × explainer × attribute) combinations exceed "
            f"the FD > {FD_THRESHOLD} threshold: "
            + "; ".join(f"{r.metric}-{r.explainer} on {r.attribute} (FD={r.FD:.3f}, "
                        f"most underserved: {r.EDI_group})" for r in flagged.itertuples())
        )
    else:
        st.success(f"No combination exceeds the FD > {FD_THRESHOLD} concern threshold at k = {k}.")

    if len(expls) == 2:
        st.markdown("#### SHAP vs LIME — which is more equitable?")
        verdicts = []
        for attr in ATTRIBUTES:
            wins = {"SHAP": 0, "LIME": 0}
            for m in ["PGI", "PGU"]:
                fds = disp[(disp.attribute == attr) & (disp.metric == m)].set_index("explainer")["FD"]
                if fds["SHAP"] < fds["LIME"]:
                    wins["SHAP"] += 1
                elif fds["LIME"] < fds["SHAP"]:
                    wins["LIME"] += 1
            verdict = ("SHAP" if wins["SHAP"] > wins["LIME"]
                       else "LIME" if wins["LIME"] > wins["SHAP"] else "tie")
            verdicts.append({"attribute": attr, "FD wins SHAP": wins["SHAP"],
                             "FD wins LIME": wins["LIME"], "more equitable": verdict})
        st.dataframe(pd.DataFrame(verdicts), hide_index=True)

    with st.expander("Full disparity table"):
        st.dataframe(
            disp[["attribute", "metric", "explainer", "global", "WSF", "WSF_group",
                  "FD", "FD_flag", "EDI", "EDI_group"]]
            .style.format({c: "{:.4f}" for c in ["global", "WSF", "FD", "EDI"]}),
            hide_index=True,
        )
