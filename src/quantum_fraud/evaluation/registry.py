"""Immutable, auditable configuration/result snapshots of saved experiments.

Historical metadata is normalized without rewriting original experiment outputs.
Unknown historical durations remain explicitly missing, never estimated as zero.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4
import numpy as np
import pandas as pd
from quantum_fraud.dashboard.research import ROOT, artifact_stem, catalog
from quantum_fraud.evaluation.metrics import binary_metrics, select_f1_threshold, select_fpr_threshold

FEATURE_RUN = "features_20260915T140754_40734f02b041"
DIAGNOSTIC = "qsvc_20260915T143124_071ab7bb9001"
PREPARED = ROOT / "data/processed/preprocessing_20260915T135023_4073b139cc42/manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_timestamp(run_id):
    match = re.search(r"_(\d{8}T\d{6})(?:_|$)", run_id)
    if not match:
        raise ValueError("Run ID does not contain an original UTC timestamp")
    return datetime.strptime(match[1], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).isoformat()


def model_settings(family, record, report, manifest):
    """Saved explicit settings plus disclosed reconstruction of factory constants."""
    settings = dict(record.get("hyperparameters", {}))
    if family == "classical":
        settings.update(seed=manifest["seed"], class_weights=manifest["effective_training_class_weights"])
        if record["model"] == "xgboost":
            settings.update(objective="binary:logistic", tree_method="hist", class_weight_application="scale_pos_weight=weight1/weight0", n_jobs=2)
        if record["model"] == "linear_svm":
            settings["dual"] = False
        if record["model"] == "random_forest":
            settings["n_jobs"] = 2
    elif family == "features":
        settings.update(C=1, max_iter=10000, dual=False, class_weights=manifest["effective_training_class_weights"])
    elif family in ("qsvc", "ablation"):
        settings.update(C=1, kernel="fidelity_statevector" if record.get("circuit") else "rbf")
        if not record.get("circuit"):
            settings["gamma"] = "scale"
    elif family == "vqc":
        settings.update(optimizer="COBYLA", max_evaluations=record["max_evaluations"], tol=1e-4,
                        initial_point="uniform(-pi,pi), seeded", loss="cross_entropy")
    elif family == "hybrid":
        settings.update(report["config"])
        settings.update(optimizer="Adam", loss="BCEWithLogitsLoss", selected_epoch=record["selected_epoch"])
    elif family == "ensemble":
        settings.update(report["config"])
        settings.update(alpha=record["alpha"], calibrator={"model": "LogisticRegression", "C": 1, "max_iter": 1000, "random_state": 42})
    elif family == "noise":
        settings.update(report["config"])
        settings.update(mode=record["model"], shots=record["shots"], frozen_source_model=report["source_model"])
    return settings


def normalize(path, report, record, base_manifest):
    family = path.name.split("_")[0]
    feature_control = family == "features"
    manifest_path = Path(report.get("subset_manifest", report.get("prepared_manifest", report.get("preprocessing_manifest", PREPARED))))
    manifest = read_json(manifest_path)
    stem = f'{record["method"]}_{record["dimension"]}' if feature_control else artifact_stem(record)
    identity = path.name + "__" + stem
    sizes = {split: {"rows": item["rows"], "fraud": item.get("fraud", item.get("class_counts", {}).get("1"))} for split, item in manifest["splits"].items()}
    if family == "ablation":
        sizes["train"] = {"rows": report["config"]["training_rows"], "fraud": 48}
    seed = record.get("seed", report.get("seed", report.get("config", {}).get("seed", manifest.get("seed", manifest.get("config", {}).get("seed")))))
    inputs = manifest.get("input_features", manifest.get("feature_names"))
    dim = record["dimension"] if feature_control else record.get("features", len(inputs))
    representation = (record["method"] if feature_control else "eligible_standardized_onehot" if family in ("classical", "hybrid") else manifest.get("config", {}).get("angle_encoding", "see_parent_manifest"))
    quantum = {key: record[key] for key in ("circuit", "feature_map", "repetitions", "ansatz_repetitions", "entanglement", "encoding_repetitions", "qubits", "latent_features", "shots", "simulation") if key in record}
    if family == "vqc":
        quantum.update(feature_map="ZZ linear, repetitions=1", ansatz="real_amplitudes", shots=None, interpretation="integer outcome modulo 2")
    if family == "qsvc" and record.get("circuit"):
        quantum.update(entanglement="linear" if record["feature_map"] == "zz" else "none", shots=None)
    if family == "noise":
        quantum.update(circuit=report["circuit"], mode=record["model"], noise=report["config"]["noise"] if record["model"] == "noisy" else None)
    if family == "hybrid" and record.get("qubits"):
        quantum.update(feature_map="ZZ linear, repetitions=1", ansatz="real_amplitudes, repetitions=1, linear", observables=["ZI", "IZ"], shots=None)
    if family == "ensemble":
        quantum["parent_models"] = report["parents"]
    config = {
        "schema_version": 1, "experiment_id": identity, "run_id": path.name,
        "timestamp_utc": run_timestamp(path.name), "timestamp_provenance": "original run ID; run start, not per-model start",
        "dataset": manifest["dataset"], "sample_size": sizes,
        "feature_set": {"eligible_encoded_inputs": inputs, "dimension": dim, "representation": representation,
                        "selected_features": record.get("features") if feature_control else None},
        "split_strategy": {"method": "whole-step temporal partitions, inherited unchanged",
                           "boundaries": {s: [m["step_min"], m["step_max"]] for s, m in base_manifest["splits"].items()},
                           "subset_protocol": manifest.get("config"), "validation_role": manifest.get("validation_role", "threshold/model selection"),
                           "training_resample": report.get("config") if family == "ablation" else None,
                           "calibration_rows": 1024 if family == "ensemble" else None},
        "random_seed": seed, "seed_role": "simulator seed; exact reference deterministic" if family == "noise" else "saved training/sampling seed",
        "preprocessing": {"manifest": relative(manifest_path), "manifest_sha256": sha256(manifest_path),
                          "base_manifest": relative(PREPARED), "policy": base_manifest["feature_policy"],
                          "method": "train-only numeric imputation/scaling, log amount, one-hot types; optional train-fitted reduction/angles as parent manifest"},
        "model": f'linear_svm_{record["method"]}' if feature_control else record["model"],
        "hyperparameters": model_settings(family, record, report, manifest), "quantum_circuit_configuration": quantum or None,
        "run_settings": report.get("config", report.get("settings")),
        "metadata_provenance": "Retrospective normalization of immutable saved reports/manifests; unrecorded fixed factory constants reconstructed from audited source snapshot. Library defaults remain version-dependent.",
        "versions": report.get("versions"), "versions_missing_reason": None if "versions" in report else "Historical feature report did not capture its runtime versions; preprocessing versions describe an earlier run.",
        "source_report": relative(path / "results.json"), "source_report_sha256": sha256(path / "results.json"),
        "selection_status": "diagnostic_excluded_from_primary_tables" if path.name == DIAGNOSTIC else "reviewed",
    }
    inference = record.get("prediction_batch_seconds")
    result = {
        "schema_version": 1, "experiment_id": identity,
        "metrics": {k: v for k, v in record.items() if k in ("validation", "test", "validation_selected_fpr_points", "validation_selected_fpr_operating_points", "fixed_ideal_threshold_test", "change_from_exact_fixed_threshold")},
        "training_duration": {"seconds": record.get("control_training_seconds") if feature_control else record.get("training_seconds"),
                              "scope": "incremental calibration/new fit only; reused parent training excluded" if family == "ensemble" else "zero additional fit; frozen parent weights" if family == "noise" else "saved fit wall time; hybrid includes validation monitoring"},
        "inference_duration": {"seconds_by_partition": inference,
                               "missing_reason": "Historical phase did not separately time model inference; not reconstructed from evaluation or blending durations" if inference is None else None,
                               "scope": "batch wall time; excludes initial preprocessing; nested simulator timings must not be added again"},
        "timing_details": {k: v for k, v in record.items() if "seconds" in k or k == "kernel_timings"},
        "source_scores": relative(path / f"{stem}_scores.npz"),
        "source_labels": {s: relative(manifest_path.parent / f"{s}_y.npy") for s in ("validation", "test")},
        "score_keys": {s: f"{s}_scores" if family == "classical" else s for s in ("validation", "test")},
        "limitations": report.get("limitations", []),
    }
    return config, result


def verify_metrics(result, root=ROOT):
    with np.load(root / result["source_scores"], allow_pickle=False) as scores:
        labels = {s: np.load(root / p, mmap_mode="r", allow_pickle=False) for s, p in result["source_labels"].items()}
        threshold = select_f1_threshold(labels["validation"], scores[result["score_keys"]["validation"]])
        np.testing.assert_allclose(threshold, result["metrics"]["validation"]["threshold"], rtol=1e-10, atol=1e-12)
        for split in ("validation", "test"):
            actual = binary_metrics(labels[split], scores[result["score_keys"][split]], threshold)
            for key, value in actual.items():
                np.testing.assert_allclose(value, result["metrics"][split][key], rtol=1e-10, atol=1e-12, err_msg=f'{result["experiment_id"]}/{split}/{key}')
        points = result["metrics"].get("validation_selected_fpr_points", result["metrics"].get("validation_selected_fpr_operating_points", {}))
        for target, point in points.items():
            cut = select_fpr_threshold(labels["validation"], scores[result["score_keys"]["validation"]], float(target))
            for split in ("validation", "test"):
                actual = binary_metrics(labels[split], scores[result["score_keys"][split]], cut)
                for key, value in actual.items():
                    np.testing.assert_allclose(value, point[split][key], rtol=1e-10, atol=1e-12)
        if "fixed_ideal_threshold_test" in result["metrics"]:
            fixed = result["metrics"]["fixed_ideal_threshold_test"]
            actual = binary_metrics(labels["test"], scores[result["score_keys"]["test"]], fixed["threshold"])
            for key, value in actual.items():
                np.testing.assert_allclose(value, fixed[key], rtol=1e-10, atol=1e-12)


def validate_registry(path, root=ROOT):
    report = read_json(path / "registry.json")
    if report["status"] != "complete":
        raise ValueError("Registry is incomplete")
    for item in report["artifacts"]:
        if sha256(root / item["path"]) != item["sha256"]:
            raise ValueError(f'Artifact hash mismatch: {item["path"]}')
    ids = set()
    for entry in report["experiments"]:
        config = read_json(root / entry["config"])
        result = read_json(root / entry["result"])
        if config["experiment_id"] in ids or config["experiment_id"] != result["experiment_id"]:
            raise ValueError("Duplicate or mismatched experiment identity")
        ids.add(config["experiment_id"])
        if result["config_sha256"] != sha256(root / entry["config"]):
            raise ValueError("Configuration hash mismatch")
        verify_metrics(result, root)
    return {"status": "complete", "experiments_verified": len(ids), "artifact_hashes_verified": len(report["artifacts"])}


def publication_outputs(configs, records, output):
    rows = []
    for config, record in zip(configs, records):
        if config["selection_status"] != "reviewed":
            continue
        test = record["metrics"]["test"]
        rows.append({"run_id": config["run_id"], "model": config["model"], "experiment_id": config["experiment_id"],
                     "features": config["feature_set"]["dimension"], "seed": config["random_seed"],
                     "train_rows": config["sample_size"]["train"]["rows"], "test_rows": config["sample_size"]["test"]["rows"],
                     "test_fraud": config["sample_size"]["test"]["fraud"],
                     **{k: v for k, v in test.items() if k != "confusion_matrix"},
                     "training_seconds": record["training_duration"]["seconds"],
                     "test_batch_seconds": (record["inference_duration"]["seconds_by_partition"] or {}).get("test")})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "all_reviewed_models.csv", index=False)
    for run, group in frame.groupby("run_id", sort=False):
        cols = ["model", "features", "seed", "pr_auc_average_precision", "roc_auc", "precision", "recall", "f1", "training_seconds", "test_batch_seconds"]
        group[cols].to_csv(output / f"{run}_table.csv", index=False)
        group[cols].to_latex(output / f"{run}_table.tex", index=False, float_format="%.5f", escape=True, caption="Fixed test results; validation-selected thresholds. Compare only under the documented run protocol.")
    repeated = frame[frame.run_id.str.startswith(("ablation_", "noise_")) & frame.seed.notna()]
    summary = repeated.groupby(["run_id", "model", "features"], sort=False).agg(
        n=("seed", "count"), ap_mean=("pr_auc_average_precision", "mean"), ap_sd=("pr_auc_average_precision", "std"),
        f1_mean=("f1", "mean"), f1_sd=("f1", "std"), training_mean=("training_seconds", "mean"), training_sd=("training_seconds", "std"),
        inference_mean=("test_batch_seconds", "mean"), inference_sd=("test_batch_seconds", "std")).reset_index()
    summary.to_csv(output / "repeated_mean_sd.csv", index=False)
    summary.to_latex(output / "repeated_mean_sd.tex", index=False, float_format="%.5f", escape=True,
                     caption="Mean and sample SD, three seeds. Ablations vary training resamples; noise varies simulator seeds, not training.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, prefix, title in zip(axes, ("ablation_", "noise_"), ("Training resamples (n=3)", "Simulator seeds (n=3)")):
        subset = summary[summary.run_id.str.startswith(prefix)]
        y = np.arange(len(subset))
        ax.errorbar(subset.ap_mean, y, xerr=subset.ap_sd, fmt="o", capsize=3, color="#2166ac")
        ax.set_yticks(y, subset.model)
        ax.set(xlabel="Test average precision (mean +/- sample SD)", title=title)
        ax.grid(axis="x", alpha=.2)
    fig.savefig(output / "repeated_studies.png", dpi=300)
    fig.savefig(output / "repeated_studies.pdf")
    plt.close(fig)


def build_registry():
    registry_id = datetime.now(timezone.utc).strftime("registry_%Y%m%dT%H%M%S_") + uuid4().hex[:8]
    output = ROOT / "results/experiment_logs" / registry_id
    config_dir = ROOT / "experiments" / registry_id
    publication = ROOT / "paper/tables" / registry_id
    for directory in (output, config_dir, publication):
        directory.mkdir(parents=True, exist_ok=False)
    base = read_json(PREPARED)
    raw = Path(base["dataset"]["path"])
    raw_hash = sha256(raw)
    if raw_hash != base["dataset"]["sha256"] or raw.stat().st_size != base["dataset"]["bytes"]:
        raise ValueError("Raw dataset changed since the preprocessing run")
    run_ids = [item for group in catalog().values() for item in group] + [FEATURE_RUN, DIAGNOSTIC]
    configs, records, entries, paths = [], [], [], {PREPARED, ROOT / "requirements.txt", ROOT / "pyproject.toml", ROOT / "configs/research_runs.json"}
    # The source snapshot is current audited code, not a claim of historical Git provenance.
    paths.update((ROOT / "src/quantum_fraud").rglob("*.py"))
    for pattern in ("*.joblib", "*_positions.npy", "*_y.npy"):
        paths.update(PREPARED.parent.glob(pattern))
    for run_id in run_ids:
        path = ROOT / "results/metrics" / run_id
        report = read_json(path / "results.json")
        if report["status"] != "complete":
            raise ValueError(f"Cannot register incomplete run {run_id}")
        paths.update(p for p in path.rglob("*") if p.is_file())
        for record in report.get("models", report.get("controls", [])):
            config, result = normalize(path, report, record, base)
            verify_metrics(result)
            config_path = config_dir / (config["experiment_id"] + ".json")
            result_path = output / (config["experiment_id"] + ".json")
            write_new(config_path, config)
            result["config"] = relative(config_path)
            result["config_sha256"] = sha256(config_path)
            write_new(result_path, result)
            configs.append(config)
            records.append(result)
            entries.append({"experiment_id": config["experiment_id"], "config": relative(config_path), "result": relative(result_path)})
            paths.update([config_path, result_path, ROOT / config["preprocessing"]["manifest"], ROOT / result["source_scores"]])
            paths.update(ROOT / p for p in result["source_labels"].values())
    publication_outputs(configs, records, publication)
    paths.update(p for p in publication.iterdir() if p.is_file())
    inventory = []
    for path in sorted((ROOT / "results/metrics").iterdir()):
        if path.is_dir():
            inventory.append({"path": relative(path), "status": "diagnostic" if path.name == DIAGNOSTIC else "registered" if path.name in run_ids else "ancillary_or_incomplete_excluded", "has_results_json": (path / "results.json").exists()})
    hardware = sorted((ROOT / "results/hardware").glob("*/request.json"))
    hardware_results = sorted((ROOT / "results/hardware").glob("*/hardware_result.json"))
    paths.update(hardware + hardware_results)
    report = {"schema_version": 1, "status": "complete", "registry_id": registry_id,
              "created_utc": datetime.now(timezone.utc).isoformat(), "experiments": entries,
              "reviewed_model_count": sum(c["selection_status"] == "reviewed" for c in configs),
              "diagnostic_model_count": sum(c["selection_status"] != "reviewed" for c in configs),
              "publication_directory": relative(publication), "run_inventory": inventory,
              "raw_dataset_verification": {"sha256": raw_hash, "bytes": raw.stat().st_size, "matches_original": True, "method": "streamed hash, no dataframe loaded"},
              "hardware": {"status": "results_available" if hardware_results else "physical_execution_pending", "requests": [relative(p) for p in hardware], "results": [relative(p) for p in hardware_results], "executed": bool(hardware_results)},
              "artifacts": [{"path": relative(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(paths)],
              "limitations": ["Retrospective normalized metadata; current audited source hashes are not historical Git hashes",
                              "Historical feature/ensemble pure inference durations are missing and explicitly null",
                              "Repeated SD is descriptive; seeds share fixed evaluation rows, not independent population studies",
                              "No statistical superiority, production readiness or quantum speedup claim",
                              "Raw CSV revalidated with a streamed hash; raw contents never changed"]}
    write_new(output / "registry.json", report)
    verification = validate_registry(output)
    write_new(output / "verification.json", verification)
    return output
