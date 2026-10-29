"""EDF → EEGraph → Knowledge Graph pipeline."""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")

import networkx as nx
import mne
from eegraph import Graph as EEG
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eegraph_ext.kg import EEGraphKGExporter
from rdflib import ConjunctiveGraph


def build_kg_from_edf(
    edf_path: str,
    window_size: int = 60,
    threshold: float = 0.7,
    montage: str = "auto",
    output_turtle: str | None = None,
    connectivity: str = "pearson_correlation",
    bands: list | None = None,
    recording_condition: str | None = None,
    source_name: str | None = None,
    segment_start: float | None = None,
    segment_end: float | None = None,
    dataset_url: str | None = None,
) -> tuple[ConjunctiveGraph, dict, list]:
    G = EEG()
    G.load_data(path=edf_path)
    _raw_info = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    _sfreq = float(_raw_info.info["sfreq"])
    _raw_info.close()

    # EEGraph derives node names by splitting channel labels on hyphens, so
    # bipolar channels sharing a pole (e.g. FP1-F7 and FP1-F3 both → FP1)
    # collapse onto the same node, producing self-loops and overwritten weights.
    # Replace hyphens with underscores before modelate so every channel keeps
    # a unique node name; translate back to original labels afterwards.
    original_ch_names = list(G.ch_names)
    G.ch_names = [c.replace("-", "_") for c in original_ch_names]
    # node_to_label maps the underscored node name EEGraph produces back to the
    # original bipolar label (e.g. "FP1_F7" → "FP1-F7").
    node_to_label = {safe.split()[-1]: orig
                     for safe, orig in zip(G.ch_names, original_ch_names)}
    assert len(node_to_label) == len(original_ch_names), \
        "channel names are not unique after hyphen→underscore renaming"

    modelate_kwargs: dict = dict(
        window_size=window_size,
        connectivity=connectivity,
        threshold=threshold,
    )
    if bands is not None:
        modelate_kwargs["bands"] = bands

    graphs_dict, raw_matrices = G.modelate(**modelate_kwargs)

    # Validate: every window must have exactly the expected number of nodes
    # and no self-loops (which would indicate pole-collision was not fixed).
    for win_idx, g in graphs_dict.items():
        assert g.number_of_nodes() == len(original_ch_names), (
            f"Window {win_idx}: expected {len(original_ch_names)} nodes, "
            f"got {g.number_of_nodes()}"
        )
        assert nx.number_of_selfloops(g) == 0, (
            f"Window {win_idx}: self-loops detected — bipolar pole collision not resolved"
        )

    # Rename nodes in every graph back to original labels so the exporter
    # receives the canonical bipolar labels it needs for IRI and pole construction.
    graphs_dict = {
        win_idx: nx.relabel_nodes(g, node_to_label)
        for win_idx, g in graphs_dict.items()
    }

    exp = EEGraphKGExporter(
        graphs_dict=graphs_dict,
        ch_names=original_ch_names,
        connectivity=connectivity,
        threshold=threshold,
        eeg_file=source_name if source_name else os.path.basename(edf_path),
        window_size=window_size,
        montage=montage,
        bands=bands,
        recording_condition=recording_condition,
        sample_rate=_sfreq,
        segment_start=segment_start,
        segment_end=segment_end,
        dataset_url=dataset_url,
    )
    cg = exp.build()
    if output_turtle:
        os.makedirs(os.path.dirname(os.path.abspath(output_turtle)), exist_ok=True)
        exp.to_turtle(output_turtle)
    return cg, graphs_dict, original_ch_names


def build_kg_psd(
    edf_path: str,
    window_size: int = 60,
    threshold: float = 0.25,   # top-25% channels retained
    bands: str = "delta, theta, alpha, beta, gamma",
    output_turtle: str | None = None,
) -> tuple[ConjunctiveGraph, dict, list]:
    """Run Power Spectrum (single-channel, band-resolved) and export to KG."""
    G = EEG()
    G.load_data(path=edf_path)
    original_ch_names = list(G.ch_names)

    graphs_dict, _ = G.modelate(
        window_size=window_size,
        connectivity="power_spectrum",
        threshold=threshold,
        bands=bands,
    )

    active_bands = [b.strip() for b in bands.split(",")]

    exp = EEGraphKGExporter(
        graphs_dict=graphs_dict,
        ch_names=original_ch_names,
        connectivity="power_spectrum",
        threshold=threshold,
        eeg_file=os.path.basename(edf_path),
        window_size=window_size,
        montage="auto",
        bands=active_bands,
    )
    cg = exp.build()
    if output_turtle:
        os.makedirs(os.path.dirname(os.path.abspath(output_turtle)), exist_ok=True)
        exp.to_turtle(output_turtle)
    return cg, graphs_dict, original_ch_names


def build_kg_granger(
    edf_path: str,
    window_size: int = 60,
    threshold: float = 0.1,   # F-test p-value threshold
    output_turtle: str | None = None,
) -> tuple[ConjunctiveGraph, dict, list]:
    """Run Granger Causality (directed, broadband) and export to KG."""
    G = EEG()
    G.load_data(path=edf_path)
    original_ch_names = list(G.ch_names)

    # Granger uses bipolar pole names as nodes — rename hyphens to underscores
    original_ch_names = list(G.ch_names)
    G.ch_names = [c.replace("-", "_") for c in original_ch_names]
    node_to_label = {safe.split()[-1]: orig
                     for safe, orig in zip(G.ch_names, original_ch_names)}

    graphs_dict, _ = G.modelate(
        window_size=window_size,
        connectivity="granger_causality",
        threshold=threshold,
    )

    # Relabel nodes back to original channel names
    graphs_dict = {
        win_idx: nx.relabel_nodes(g, node_to_label)
        for win_idx, g in graphs_dict.items()
    }

    exp = EEGraphKGExporter(
        graphs_dict=graphs_dict,
        ch_names=original_ch_names,
        connectivity="granger_causality",
        threshold=threshold,
        eeg_file=os.path.basename(edf_path),
        window_size=window_size,
        montage="auto",
    )
    cg = exp.build()
    if output_turtle:
        os.makedirs(os.path.dirname(os.path.abspath(output_turtle)), exist_ok=True)
        exp.to_turtle(output_turtle)
    return cg, graphs_dict, original_ch_names


def build_kg_from_raws(
    raw_paths: list[str],
    window_size: int = 30,
    threshold: float = 0.7,
    montage: str = "monopolar",
    output_turtle: str | None = None,
    recording_condition: str | None = None,
    dataset_url: str | None = None,
) -> tuple[ConjunctiveGraph, dict, list]:
    import tempfile
    # Derive a canonical provenance name from the source files
    basenames = [os.path.basename(p) for p in raw_paths]
    canonical_name = "+".join(basenames) if len(basenames) > 1 else basenames[0]
    raws = [mne.io.read_raw_edf(p, preload=True, verbose=False) for p in raw_paths]
    raw = mne.concatenate_raws(raws)
    raw.rename_channels({ch: ch.rstrip(".").upper() for ch in raw.ch_names})
    # sfreq is preserved in the temp EDF exported below and read by build_kg_from_edf
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp_path = tmp.name
    mne.export.export_raw(tmp_path, raw, fmt="edf", overwrite=True, verbose=False)
    try:
        cg, graphs_dict, ch_names = build_kg_from_edf(
            tmp_path, window_size, threshold, montage, output_turtle,
            recording_condition=recording_condition, source_name=canonical_name,
            dataset_url=dataset_url)
    finally:
        os.unlink(tmp_path)
    return cg, graphs_dict, ch_names


def parse_chbmit_summary(summary_path: str) -> dict[str, list[tuple[int, int]]]:
    """Parse a chbXX-summary.txt file.
    Returns {filename: [(onset_sec, offset_sec), ...]} for files with seizures.
    """
    seizures: dict[str, list[tuple[int, int]]] = {}
    current_file = None
    with open(summary_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("File Name:"):
                current_file = line.split(":", 1)[1].strip()
                seizures.setdefault(current_file, [])
            elif line.startswith("Seizure Start Time:") and current_file:
                onset = int(line.split(":")[1].strip().split()[0])
            elif line.startswith("Seizure End Time:") and current_file:
                offset = int(line.split(":")[1].strip().split()[0])
                seizures[current_file].append((onset, offset))
    return {k: v for k, v in seizures.items() if v}


def build_kg_from_edf_segment(
    edf_path: str,
    t_start: float,
    t_stop: float,
    window_size: int = 5,
    threshold: float = 0.45,
    montage: str = "auto",
    output_turtle: str | None = None,
    connectivity: str = "wpli",
    bands: list | None = None,
    recording_condition: str | None = None,
    dataset_url: str | None = None,
) -> tuple[ConjunctiveGraph, dict, list]:
    """Build a KG from a time-cropped segment of an EDF file."""
    import tempfile
    source_name = os.path.basename(edf_path)
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    _sfreq_seg = float(raw.info["sfreq"])
    raw.crop(tmin=t_start, tmax=t_stop)
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp_path = tmp.name
    mne.export.export_raw(tmp_path, raw, fmt="edf", overwrite=True, verbose=False)
    try:
        cg, graphs_dict, ch_names = build_kg_from_edf(
            tmp_path, window_size, threshold, montage, output_turtle,
            connectivity=connectivity, bands=bands,
            recording_condition=recording_condition, source_name=source_name,
            segment_start=t_start, segment_end=t_stop, dataset_url=dataset_url)
    finally:
        os.unlink(tmp_path)
    return cg, graphs_dict, ch_names
