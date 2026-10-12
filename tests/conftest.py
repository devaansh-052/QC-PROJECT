"""Use the same native-runtime bootstrap as project entry points on Windows."""

# Selected test files may import sklearn directly before importing package code.
# Bootstrap here too; the fresh-process regression separately checks real clients.
import quantum_fraud  # noqa: F401
