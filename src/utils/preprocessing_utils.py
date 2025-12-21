import numpy as np
import pandas as pd


MISSING_MARKERS = {"?", "na", "n/a", "null", "none", "", "nan", "missing"}


def normalize_missing_and_strip(df: pd.DataFrame) -> pd.DataFrame:
	"""Normalize missing markers and strip whitespace from object/category columns.

	- Strips leading/trailing whitespace on string-like columns.
	- Replaces common missing tokens with np.nan (case-insensitive) after strip.
	- Leaves numeric columns untouched.
	"""
	cleaned = df.copy()
	for col in cleaned.columns:
		if cleaned[col].dtype == object or pd.api.types.is_categorical_dtype(cleaned[col]):
			# Strip whitespace
			cleaned[col] = cleaned[col].astype(str).str.strip()
			# Normalize missing tokens
			cleaned[col] = cleaned[col].apply(
				lambda x: np.nan if str(x).strip().lower() in MISSING_MARKERS else x
			)
	return cleaned
