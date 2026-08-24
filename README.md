# OnBase Percentage Prediction

XGBoost-based professional-baseball probability prediction system. Deploy
`streamlit_app.py` as the Streamlit entry point. One website contains two pages:

- Plate-appearance final on-base probability
- Contact-instant hit probability

## Models

- **Plate-appearance model:** before each pitch, estimates the probability that
  the plate appearance ultimately ends with the batter reaching base.
- **Contact-instant model:** after contact, estimates hit probability using only
  information available at contact. Final landing coordinates, actual flight
  distance, and post-play batted-ball classifications are excluded.

The two scores are not directly comparable because their targets and information
cutoffs differ. Both interfaces provide local TreeSHAP explanations.

## Local run

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Raw Statcast data, training reports, training-only outputs, and credentials are
intentionally excluded from this deployment repository.
