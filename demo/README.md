# Live demo
## Run it

```bash
pip install streamlit shap lime scikit-learn pandas numpy altair tqdm

# one-time: compute SHAP + LIME attributions on the 500-instance sample
python demo/precompute.py

streamlit run demo/app.py
```

Opens at 

http://localhost:8501

## Suggested flow

1. **Tab ① — Explain one person.** Pick someone (or "Surprise me"), show the live SHAP
   explanation, then run the masking test: masking the top-5 "important" features crashes
   the prediction, masking the bottom-5 barely moves it. That *is* faithfulness (PGI/PGU),
   demonstrated on one human being.
2. **Tab ② — The aggregation problem.** Show PGI globally (dashed line = the number a
   standard evaluation would report), then break it down by sex → race → age group.
   The audience sees the bars fall away from the global line for specific groups.
3. **Tab ③ — WSF · FD · EDI.** The three novel metrics turn that visual gap into numbers:
   which group is worst-served (WSF), how wide the gap is (FD, red line = 0.1 concern
   threshold), and who is most abandoned relative to the global average (EDI). Finish
   with the SHAP-vs-LIME equity verdict table.

The **k slider** in the sidebar re-runs the whole analysis live for a different number of
masked features — a nice "this is real, not slides" moment.

## Files

- `precompute.py` — rebuilds the stratified 500-instance sample and saves
  `artifacts/{sample_indices,shap_values,lime_values}.npy`
- `app.py` — the Streamlit app (loads `data/` + `demo/artifacts/`)
