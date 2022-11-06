from pathlib import Path

MAXIMUM_PREDICTION_FREQ = 0.75  # Predictions/Second
SAMPLE_OVERLAP = 0.5  # Final seconds of current sample to be used in the next sample to prevent losing speech segments
SAVE_DIR = Path(__file__).parent.resolve() / 'outputs'  # Where to save predictions and audio files (or load from)
