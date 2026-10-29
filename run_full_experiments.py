"""
Full experiment pipeline for the EEGraph-as-KG paper.

Runs all experiments for both datasets and writes outputs to outputs/.
Run from the EEGraph-KG-Paper directory:

    python run_full_experiments.py [--single-subject-only]

--single-subject-only  Skip population-scale (133 subjects) experiments.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from src.kg_pipeline    import (build_kg_from_raws,
                                 build_kg_from_edf_segment, parse_chbmit_summary)
from src.kg_experiments import (characterise_kg, run_sparql_queries,
                                 compute_graph_metrics, jaccard_per_window,
                                 jaccard_summary, check_prov_completeness,
                                 fair_evaluation)

BASE    = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE, "outputs")
KG_DIR  = os.path.join(OUT_DIR, "kg")
os.makedirs(KG_DIR, exist_ok=True)

# ── dataset paths ─────────────────────────────────────────────────────────────
CHB_NONSZ_EDF   = os.path.join(BASE, "chbmit_full", "chb01", "chb01_01.edf")
CHB_SEIZURE_EDF = os.path.join(BASE, "chbmit_full", "chb01", "chb01_03.edf")
CHB_SUMMARY     = os.path.join(BASE, "chbmit_full", "chb01", "chb01-summary.txt")
EEG_REST_EDFS = sorted(glob.glob(os.path.join(BASE, "eegmmidb_full", "S001", "*R01*.edf")) +
                        glob.glob(os.path.join(BASE, "eegmmidb_full", "S001", "*R02*.edf")))
EEG_TASK_EDFS = sorted(glob.glob(os.path.join(BASE, "eegmmidb_full", "S001", "*R04*.edf")) +
                        glob.glob(os.path.join(BASE, "eegmmidb_full", "S001", "*R08*.edf")) +
                        glob.glob(os.path.join(BASE, "eegmmidb_full", "S001", "*R12*.edf")))

# ── helpers ───────────────────────────────────────────────────────────────────

def _save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def _header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ── 5.2  KG characterisation ──────────────────────────────────────────────────

def run_kg_characterisation(args):
    _header("5.2  KG Characterisation (single subject)")
    results = {}

    for label, edfs, ttl, condition in [
        ("S001_rest", EEG_REST_EDFS, os.path.join(KG_DIR, "S001_rest.trig"), "RestingState"),
        ("S001_task", EEG_TASK_EDFS, os.path.join(KG_DIR, "S001_task.trig"), "TaskRelated"),
    ]:
        if not edfs:
            print(f"  SKIP {label}: no EDFs found"); continue
        print(f"  Building {label} KG ({len(edfs)} files) …")
        cg, _, _ = build_kg_from_raws(edfs, window_size=30, threshold=0.7,
                                       montage="monopolar", output_turtle=ttl,
                                       recording_condition=condition,
                                       dataset_url="https://doi.org/10.13026/C28G6P")
        stats = characterise_kg(cg)
        results[label] = stats
        print(f"  {label}: {stats}")

    _save_json(results, os.path.join(OUT_DIR, "kg_characterisation.json"))


# ── 5.3  Analytical SPARQL queries ───────────────────────────────────────────

def run_sparql_section():
    from rdflib import ConjunctiveGraph
    _header("5.3  Analytical SPARQL Queries")
    results = {}
    for label, ttl in [
        ("chb01_seizure", os.path.join(KG_DIR, "chb01_seizure.trig")),
        ("S001_rest",     os.path.join(KG_DIR, "S001_rest.trig")),
    ]:
        if not os.path.exists(ttl):
            print(f"  SKIP {label}: {ttl} not found"); continue
        cg = ConjunctiveGraph(); cg.parse(ttl, format="trig")
        r = run_sparql_queries(cg)
        results[label] = r
        print(f"  {label}: Q1={r.get('Q1_nifstd_mapped')}  "
              f"Q4_total={r.get('Q_total_edges')}")
    _save_json(results, os.path.join(OUT_DIR, "sparql_queries.json"))


# ── 5.4  Single-subject graph metrics ────────────────────────────────────────

def run_single_subject_metrics():
    from rdflib import ConjunctiveGraph
    import networkx as nx, numpy as np
    _header("5.4  Single-Subject Graph Metrics")

    def _metrics_from_ttl(ttl, window_size, threshold, montage, edf=None, edfs=None):
        cg = ConjunctiveGraph()
        if os.path.exists(ttl):
            cg.parse(ttl, format="trig")
            # rebuild graphs_dict from KG edges
            from rdflib import URIRef
            EEGO = "http://eegraph.org/ontology#"
            edges_q = f"""PREFIX eego: <{EEGO}>
            SELECT ?src ?tgt ?w ?win WHERE {{
              ?e a eego:ConnectivityEdge ;
                 eego:sourceElectrode ?src ;
                 eego:targetElectrode ?tgt ;
                 eego:weight ?w ;
                 eego:inWindow ?win .
            }}"""
            from collections import defaultdict
            gd = defaultdict(nx.Graph)
            for r in cg.query(edges_q):
                win_idx = int(str(r.win).split("_")[-1])
                src = str(r.src).split("/")[-1]
                tgt = str(r.tgt).split("/")[-1]
                gd[win_idx].add_edge(src, tgt, weight=float(r.w))
            return compute_graph_metrics(dict(gd))
        return {}

    results = {}
    for label, ttl in [
        ("chb01_seizure", os.path.join(KG_DIR, "chb01_seizure.trig")),
        ("chb01_nonsz",   os.path.join(KG_DIR, "chb01_nonsz.trig")),
        ("S001_rest",     os.path.join(KG_DIR, "S001_rest.trig")),
        ("S001_task",     os.path.join(KG_DIR, "S001_task.trig")),
    ]:
        if not os.path.exists(ttl):
            print(f"  SKIP {label}: not found"); continue
        m = _metrics_from_ttl(ttl, 60, 0.7, "auto")
        results[label] = m
        print(f"  {label}: density={m.get('density', 0):.4f}  "
              f"transitivity={m.get('transitivity', 0):.4f}")
    _save_json(results, os.path.join(OUT_DIR, "single_subject_metrics.json"))


# ── 5.5  Edge overlap (Jaccard) ───────────────────────────────────────────────

def run_edge_overlap():
    _header("5.5  Edge Overlap Across Windows")
    import networkx as nx
    from collections import defaultdict
    from rdflib import ConjunctiveGraph

    def _load_gd(ttl):
        cg = ConjunctiveGraph(); cg.parse(ttl, format="trig")
        EEGO = "http://eegraph.org/ontology#"
        edges_q = f"""PREFIX eego: <{EEGO}>
        SELECT ?src ?tgt ?win WHERE {{
          ?e a eego:ConnectivityEdge ;
             eego:sourceElectrode ?src ;
             eego:targetElectrode ?tgt ;
             eego:inWindow ?win .
        }}"""
        gd = defaultdict(nx.Graph)
        for r in cg.query(edges_q):
            wi = int(str(r.win).split("_")[-1])
            gd[wi].add_edge(str(r.src).split("/")[-1], str(r.tgt).split("/")[-1])
        return dict(gd)

    results = {}
    for label, ttl_a, ttl_b in [
        ("chb01_seizure_vs_nonsz", os.path.join(KG_DIR, "chb01_seizure.trig"),
                                   os.path.join(KG_DIR, "chb01_nonsz.trig")),
        ("S001_rest_vs_task",      os.path.join(KG_DIR, "S001_rest.trig"),
                                   os.path.join(KG_DIR, "S001_task.trig")),
    ]:
        if not (os.path.exists(ttl_a) and os.path.exists(ttl_b)):
            print(f"  SKIP {label}"); continue
        gd_a = _load_gd(ttl_a); gd_b = _load_gd(ttl_b)
        scores = jaccard_per_window(gd_a, gd_b)
        summary = jaccard_summary(scores)
        results[label] = summary
        print(f"  {label}: mean={summary['mean']:.3f}  "
              f"std={summary['std']:.3f}  "
              f"min={summary['min']:.3f}  max={summary['max']:.3f}")
    _save_json(results, os.path.join(OUT_DIR, "edge_overlap.json"))


# ── 5.9  PROV-O completeness ─────────────────────────────────────────────────

def run_prov_completeness():
    from rdflib import ConjunctiveGraph
    _header("5.9  PROV-O Completeness")
    results = {}
    for label, ttl in [
        ("chb01_seizure", os.path.join(KG_DIR, "chb01_seizure.trig")),
        ("chb01_nonsz",   os.path.join(KG_DIR, "chb01_nonsz.trig")),
        ("S001_rest",     os.path.join(KG_DIR, "S001_rest.trig")),
        ("S001_task",     os.path.join(KG_DIR, "S001_task.trig")),
    ]:
        if not os.path.exists(ttl):
            print(f"  SKIP {label}"); continue
        cg = ConjunctiveGraph(); cg.parse(ttl, format="trig")
        checks = check_prov_completeness(cg)
        results[label] = checks
        n_pass = sum(checks.values())
        print(f"  {label}: {n_pass}/{len(checks)} pass  {checks}")
    _save_json(results, os.path.join(OUT_DIR, "prov_completeness.json"))


# ── 5.10  Ontology evaluation (CQs + FAIR) ────────────────────────────────────

def run_ontology_evaluation():
    from rdflib import ConjunctiveGraph
    from src.ontology_evaluation import run_cq_evaluation
    _header("5.10  Ontology Evaluation (CQs)")
    results = {}
    for label, ttl in [
        ("chb01",  os.path.join(KG_DIR, "chb01_ctrl.trig")),
        ("S001",   os.path.join(KG_DIR, "S001_rest.trig")),
    ]:
        if not os.path.exists(ttl):
            print(f"  SKIP {label}"); continue
        cg = ConjunctiveGraph(); cg.parse(ttl, format="trig")
        cq_results = run_cq_evaluation(cg)
        results[label] = cq_results
        n_pass = sum(1 for v in cq_results.values() if v)
        print(f"  {label}: {n_pass}/{len(cq_results)} CQs pass")
    _save_json(results, os.path.join(OUT_DIR, "cq_evaluation.json"))


# ── 5.11  FAIR evaluation ─────────────────────────────────────────────────────

def run_fair_evaluation():
    from rdflib import ConjunctiveGraph
    _header("5.11  FAIR Evaluation")
    results = {}
    for label, ttl in [
        ("chb01_ctrl", os.path.join(KG_DIR, "chb01_ctrl.trig")),
        ("S001_rest",  os.path.join(KG_DIR, "S001_rest.trig")),
    ]:
        if not os.path.exists(ttl):
            print(f"  SKIP {label}"); continue
        cg = ConjunctiveGraph(); cg.parse(ttl, format="trig")
        r = fair_evaluation(cg)
        results[label] = r
        print(f"  {label}: {r.get('score')}")
    _save_json(results, os.path.join(OUT_DIR, "fair_evaluation.json"))


# ── 5.11b  SHACL Validation ──────────────────────────────────────────────────

def run_shacl_validation():
    import pyshacl
    from rdflib import ConjunctiveGraph, Graph
    _header("5.11b  SHACL Validation")

    shapes_path = os.path.join(BASE, "eegraph_ext", "eego-shapes.ttl")
    if not os.path.exists(shapes_path):
        print("  SKIP: eego-shapes.ttl not found"); return

    shapes_graph = Graph()
    shapes_graph.parse(shapes_path, format="turtle")

    results = {}
    for label, ttl in [
        ("chb01_seizure", os.path.join(KG_DIR, "chb01_seizure.trig")),
        ("chb01_nonsz",   os.path.join(KG_DIR, "chb01_nonsz.trig")),
        ("S001_rest",     os.path.join(KG_DIR, "S001_rest.trig")),
        ("S001_task",     os.path.join(KG_DIR, "S001_task.trig")),
    ]:
        if not os.path.exists(ttl):
            print(f"  SKIP {label}: file not found"); continue
        cg = ConjunctiveGraph()
        cg.parse(ttl, format="trig")
        conforms, report_graph, report_text = pyshacl.validate(
            cg,
            shacl_graph=shapes_graph,
            inference="none",
            abort_on_first=False,
        )
        status = "PASS" if conforms else "FAIL"
        n_violations = report_text.count("Constraint Violation")
        print(f"  {label}: {status}  ({n_violations} violations)")
        results[label] = {"conforms": conforms, "n_violations": n_violations}
        if not conforms:
            for line in report_text.splitlines():
                if "Message:" in line or "Focus" in line:
                    print(f"    {line.strip()}")

    _save_json(results, os.path.join(OUT_DIR, "shacl_validation.json"))


# ── 5.12  Seizure vs non-seizure comparison (CHB-MIT chb01) ──────────────────

def run_seizure_comparison():
    from rdflib import Graph, Literal, URIRef
    from rdflib.namespace import RDF, RDFS, XSD
    _header("5.12  Seizure vs Non-Seizure KG Comparison")

    if not os.path.exists(CHB_SUMMARY):
        print("  SKIP: chb01-summary.txt not found"); return
    if not os.path.exists(CHB_SEIZURE_EDF):
        print("  SKIP: chb01_03.edf not found"); return
    if not os.path.exists(CHB_NONSZ_EDF):
        print("  SKIP: chb01_01.edf not found"); return

    # Parse seizure intervals from summary
    summary = parse_chbmit_summary(CHB_SUMMARY)
    seizures_03 = summary.get("chb01_03.edf", [])
    if not seizures_03:
        print("  SKIP: no seizures found in chb01_03.edf"); return

    onset, offset = seizures_03[0]           # 2996–3036 s → 40 s seizure
    window_size   = 5                        # 5 s windows → 8 seizure windows
    n_windows     = (offset - onset) // window_size
    threshold     = 0.45

    print(f"  Seizure: {onset}–{offset} s → {n_windows} × {window_size} s windows")

    # ── seizure KG ───────────────────────────────────────────────────────────
    sz_ttl = os.path.join(KG_DIR, "chb01_seizure.trig")
    print("  Building seizure KG …")
    cg_sz, _, _ = build_kg_from_edf_segment(
        CHB_SEIZURE_EDF, t_start=onset, t_stop=offset,
        window_size=window_size, threshold=threshold,
        connectivity="wpli", bands=["beta"],
        output_turtle=sz_ttl,
        recording_condition="PathologicalActivity",
        dataset_url="https://doi.org/10.13026/C28G6P",
    )

    # Add AnnotatedEvent triples to the seizure KG
    EEGO  = "http://eegraph.org/ontology#"
    EEGR  = "http://eegraph.org/recording/"
    event_uri = URIRef(f"{EEGR}chb01_03/event/seizure_0")
    meta = Graph()
    meta.add((event_uri, RDF.type,                    URIRef(f"{EEGO}AnnotatedEvent")))
    meta.add((event_uri, RDFS.label,                  Literal("seizure", datatype=XSD.string)))
    meta.add((event_uri, URIRef(f"{EEGO}eventOnset"),  Literal(onset,  datatype=XSD.float)))
    meta.add((event_uri, URIRef(f"{EEGO}eventOffset"), Literal(offset, datatype=XSD.float)))
    meta.add((event_uri, URIRef(f"{EEGO}eventLabel"),  Literal("ictal", datatype=XSD.string)))
    # Append event triples to the TTL file
    with open(sz_ttl, "a") as f:
        for s, p, o in meta:
            f.write(f"<{s}> <{p}> {o.n3()} .\n")
    stats_sz = characterise_kg(cg_sz)
    print(f"  seizure KG: {stats_sz}")

    # ── non-seizure KG ────────────────────────────────────────────────────────
    # Use midpoint of chb01_01.edf (no seizures); same duration as seizure segment
    nonsz_ttl = os.path.join(KG_DIR, "chb01_nonsz.trig")
    mid = 1800.0                             # 30 min into the 1-hour file
    nonsz_start = mid
    nonsz_stop  = mid + (offset - onset)    # same 40 s duration
    print(f"  Non-seizure: {nonsz_start}–{nonsz_stop} s from chb01_01.edf")
    print("  Building non-seizure KG …")
    cg_ns, _, _ = build_kg_from_edf_segment(
        CHB_NONSZ_EDF, t_start=nonsz_start, t_stop=nonsz_stop,
        window_size=window_size, threshold=threshold,
        connectivity="wpli", bands=["beta"],
        output_turtle=nonsz_ttl,
        recording_condition="PathologicalActivity",
        dataset_url="https://doi.org/10.13026/C28G6P",
    )
    stats_ns = characterise_kg(cg_ns)
    print(f"  non-seizure KG: {stats_ns}")

    # ── comparison ────────────────────────────────────────────────────────────
    from src.kg_experiments import run_sparql_queries
    q_sz = run_sparql_queries(cg_sz)
    q_ns = run_sparql_queries(cg_ns)

    results = {
        "seizure":     {**stats_sz, "total_edges": q_sz.get("Q_total_edges", 0)},
        "non_seizure": {**stats_ns, "total_edges": q_ns.get("Q_total_edges", 0)},
        "window_size_s": window_size,
        "n_windows":     n_windows,
        "measure":       "wpli",
        "band":          "beta",
        "threshold":     threshold,
        "seizure_onset": onset,
        "seizure_offset": offset,
    }
    _save_json(results, os.path.join(OUT_DIR, "seizure_comparison.json"))
    print(f"\n  Seizure edges:     {results['seizure']['total_edges']}")
    print(f"  Non-seizure edges: {results['non_seizure']['total_edges']}")
    dens_sz = results['seizure'].get('n_triples', 0)
    dens_ns = results['non_seizure'].get('n_triples', 0)
    print(f"  Seizure triples:     {dens_sz}")
    print(f"  Non-seizure triples: {dens_ns}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-subject-only", action="store_true")
    args = parser.parse_args()

    run_kg_characterisation(args)
    run_sparql_section()
    run_single_subject_metrics()
    run_edge_overlap()
    run_prov_completeness()
    run_ontology_evaluation()
    run_fair_evaluation()
    run_shacl_validation()
    run_seizure_comparison()

    if not args.single_subject_only:
        print("\n[Population-scale experiments: run src/population_metrics.py separately]")
        print("[Phase analysis: run src/preictal_analysis.py separately]")
        print("[Baseline comparison: run src/baseline_comparison.py separately]")
        print("[Threshold sensitivity: run src/sensitivity_analysis.py separately]")

    print("\n\nAll outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
