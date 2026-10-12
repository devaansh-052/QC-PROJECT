"""Dataset-level feature policy, defined before model experiments."""

TARGET_COLUMN = "isFraud"
BALANCE_COLUMNS = ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
PRIMARY_FEATURE_COLUMNS = ("step", "type", "amount")
EXCLUDED_FEATURE_COLUMNS = (*BALANCE_COLUMNS, "nameOrig", "nameDest", "isFlaggedFraud")


def primary_feature_policy() -> dict[str, object]:
    """Return policy metadata, not a fitted transform or a copy of dataset rows.

    Future preprocessing must select this allowlist explicitly. Selecting every
    column except the target is unsafe. Balance-derived features are also excluded.
    """
    return {
        "version": "paysim-primary-v1",
        "target": TARGET_COLUMN,
        "candidate_features": list(PRIMARY_FEATURE_COLUMNS),
        "excluded_features": list(EXCLUDED_FEATURE_COLUMNS),
        "exclude_balance_derived_features": True,
        "source": "https://www.kaggle.com/datasets/ealaxi/paysim1/metadata",
        "reason": "Publisher warns against all four balance fields because detected fraud transactions are cancelled; identifiers and existing detection flags are excluded from primary inputs.",
        "scope": "Feature eligibility only; no preprocessing or model training performed",
    }
