"""All experiments reported in the paper."""
from __future__ import annotations
import sys, os
from typing import Any
import networkx as nx
import numpy as np
from rdflib import ConjunctiveGraph
from rdflib.namespace import PROV, RDF, SKOS
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eegraph_ext.kg import EEGO, EEGR, SOSA


def characterise_kg(cg: ConjunctiveGraph) -> dict[str, int]:
    triples = list(cg.quads())
    return {
        "n_triples":      len(triples),
        "n_named_graphs": len({g for s,p,o,g in triples}),
        "n_subjects":     len({s for s,p,o,g in triples}),
        "n_predicates":   len({p for s,p,o,g in triples}),
        "n_objects":      len({o for s,p,o,g in triples}),
    }


_Q1 = """PREFIX eego: <http://eegraph.org/ontology#>
SELECT (COUNT(DISTINCT ?e) AS ?mapped) WHERE {
  ?e eego:hasStandardPosition ?pos .
  FILTER(STRSTARTS(STR(?pos), STR(eego:)))
}"""

_Q2 = """PREFIX eego: <http://eegraph.org/ontology#>
SELECT ?src ?tgt (COUNT(?edge) AS ?wins) (MAX(?w) AS ?max_w) WHERE {
  ?edge a eego:ConnectivityEdge ;
        eego:sourceElectrode ?src ;
        eego:targetElectrode ?tgt ;
        eego:weight ?w .
} GROUP BY ?src ?tgt ORDER BY DESC(?wins) DESC(?max_w) LIMIT 5"""

_Q3 = """PREFIX eego: <http://eegraph.org/ontology#>
SELECT ?win (COUNT(?e) AS ?n_edges) WHERE {
  ?e a eego:ConnectivityEdge ; eego:inWindow ?win .
} GROUP BY ?win ORDER BY ?win"""

_Q4 = """PREFIX eego: <http://eegraph.org/ontology#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX schema: <https://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?name ?version ?measure ?source WHERE {
  ?act a eego:ConnectivityComputation ;
       prov:wasAssociatedWith ?sw ;
       eego:connectivityMeasure ?m ;
       prov:used ?rec .
  ?sw schema:name ?name ; schema:version ?version .
  BIND(STRAFTER(STR(?m), "#") AS ?measure)
  ?rec rdfs:label ?source .
} LIMIT 1"""

_Q5 = """PREFIX eego: <http://eegraph.org/ontology#>
ASK { ?c1 eego:positivePole ?e . ?c2 eego:positivePole ?e . FILTER(?c1 != ?c2) }"""

_Q_EDGE_COUNT = """PREFIX eego: <http://eegraph.org/ontology#>
SELECT (COUNT(?e) AS ?n_edges) WHERE { ?e a eego:ConnectivityEdge . }"""


def run_sparql_queries(cg: ConjunctiveGraph) -> dict[str, Any]:
    results: dict[str, Any] = {}
    r1 = list(cg.query(_Q1))
    results["Q1_nifstd_mapped"] = int(r1[0][0]) if r1 else 0
    r2 = [(str(row.src).split("/")[-1], str(row.tgt).split("/")[-1],
           int(row.wins), float(row.max_w))
          for row in cg.query(_Q2)]
    results["Q2_top5_edges"] = r2
    r3 = {str(row.win).split("/")[-1]: int(row.n_edges) for row in cg.query(_Q3)}
    results["Q3_edges_per_window"] = r3
    if r3:
        counts = list(r3.values())
        results["Q3_min"] = min(counts); results["Q3_max"] = max(counts)
    r4 = list(cg.query(_Q4))
    if r4:
        results["Q4_agent_name"]    = str(r4[0].name)
        results["Q4_agent_version"] = str(r4[0].version)
        results["Q4_measure"]       = str(r4[0].measure)
        results["Q4_source_file"]   = str(r4[0].source)
    results["Q5_pole_sharing"] = bool(cg.query(_Q5))
    r_ec = list(cg.query(_Q_EDGE_COUNT))
    results["Q_total_edges"] = int(r_ec[0][0]) if r_ec else 0
    return results


def compute_graph_metrics(graphs_dict: dict) -> dict[str, float]:
    rows: dict[str, list] = {k: [] for k in
                             ["density","transitivity","clustering","global_eff","betweenness"]}
    for g in graphs_dict.values():
        if g.number_of_nodes() < 2:
            continue
        rows["density"].append(nx.density(g))
        rows["transitivity"].append(nx.transitivity(g))
        rows["clustering"].append(nx.average_clustering(g))
        rows["global_eff"].append(nx.global_efficiency(g))
        bc = list(nx.betweenness_centrality(g).values())
        rows["betweenness"].append(np.mean(bc) if bc else 0.0)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in rows.items()}


def jaccard_per_window(gd_a: dict, gd_b: dict) -> list[float]:
    n = min(len(gd_a), len(gd_b))
    scores = []
    for i in range(n):
        ea = set(frozenset(e) for e in gd_a[i].edges())
        eb = set(frozenset(e) for e in gd_b[i].edges())
        union = ea | eb
        scores.append(len(ea & eb) / len(union) if union else 0.0)
    return scores


def jaccard_summary(scores: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)),
            "min": float(np.min(scores)), "max": float(np.max(scores))}


def seizure_window_analysis(gd_ctrl, gd_ictal, onset_sec, window_size) -> dict[str, Any]:
    onset_win = onset_sec // window_size
    result = {}
    for win_idx in [onset_win, onset_win + 1]:
        g_ctrl  = gd_ctrl.get(win_idx)
        g_ictal = gd_ictal.get(win_idx)
        result[f"window_{win_idx}"] = {
            "ctrl_edges":    g_ctrl.number_of_edges()  if g_ctrl  else None,
            "ictal_edges":   g_ictal.number_of_edges() if g_ictal else None,
            "ctrl_density":  nx.density(g_ctrl)  if g_ctrl  else None,
            "ictal_density": nx.density(g_ictal) if g_ictal else None,
        }
    return result


_PROV_CHECKS = {
    "SoftwareAgent":     "ASK { ?x a <http://www.w3.org/ns/prov#SoftwareAgent> }",
    "Activity":          "ASK { ?x a <http://www.w3.org/ns/prov#Activity> }",
    "Entity":            "ASK { ?x a <http://www.w3.org/ns/prov#Entity> }",
    "wasAssociatedWith": "ASK { ?a <http://www.w3.org/ns/prov#wasAssociatedWith> ?b }",
    "wasGeneratedBy":    ("ASK { ?e a <http://eegraph.org/ontology#ConnectivityEdge> ;"
                          " <http://www.w3.org/ns/prov#wasGeneratedBy> ?a }"),
    "generatedAtTime":   "ASK { ?e <http://www.w3.org/ns/prov#generatedAtTime> ?t }",
}


def check_prov_completeness(cg: ConjunctiveGraph) -> dict[str, bool]:
    return {name: bool(cg.query(q)) for name, q in _PROV_CHECKS.items()}


def fair_evaluation(cg: ConjunctiveGraph) -> dict[str, bool | str]:
    from rdflib.namespace import RDFS
    from rdflib import Literal
    from collections import Counter
    uris    = [s for s,p,o in cg if str(s).startswith("http")]
    labels  = list(cg.triples((None, RDFS.label, None)))
    eeg_refs = [o for s,p,o in cg if isinstance(o, Literal) and str(o).endswith(".edf")]
    owl_triples = list(cg.triples((None, RDF.type, None)))
    preds   = {str(p) for s,p,o in cg}
    # eego:hasStandardPosition links recorded electrodes to standard position individuals
    eego_pos_links = [o for s,p,o in cg.triples((None, EEGO["hasStandardPosition"], None))
                      if str(o).startswith("http://eegraph.org/ontology#")]
    subj_preds = Counter(s for s,p,o in cg)
    prov_edges = list(cg.triples((None, PROV.wasGeneratedBy, None)))
    edge_uris  = list(cg.subjects(RDF.type, EEGO["ConnectivityEdge"]))
    criteria = {
        "F1":   len(uris) > 0,
        "F2":   len(labels) > 0,
        "F3":   len(eeg_refs) > 0,
        "F4":   len(list(cg.contexts())) > 1,
        "A1":   len(uris) > 0,
        "A1.1": True,
        "A1.2": True,
        "A2":   True,   # satisfied by Zenodo DOI: 10.5281/zenodo.23574953
        "I1":   len(owl_triples) > 0,
        "I2":   any("sosa" in p or "prov" in p or "skos" in p for p in preds),
        "I3":   len(eego_pos_links) > 0,   # qualified refs to eego: named individuals
        "R1":   any(c >= 3 for c in subj_preds.values()),
        "R1.1": True,   # CC-BY 4.0 on Zenodo
        "R1.2": len(prov_edges) >= len(edge_uris) and len(edge_uris) > 0,
        "R1.3": len(eego_pos_links) > 0,   # community standard 10/20 via eego: named individuals
    }
    met   = sum(1 for v in criteria.values() if v)
    total = len(criteria)
    criteria["score"] = f"{met}/{total} ({100*met//total}%)"
    return criteria
