# EEG Connectivity as a Knowledge Graph — Companion Code

This repository contains the ontology, knowledge graphs, pipeline code, and verification notebook for the paper:

> **"Representing EEG Functional Connectivity as a FAIR Knowledge Graph"**  
> Alberto Nogales, Álvaro J. García-Tejedor  
> Universidad de Alcalá (UAH) — IERU Research Group

---

## Repository structure

```
├── ontology/
│   └── eego.ttl               # EEGraph EEG Ontology (OWL, 19 classes, 29 properties, 384 individuals)
├── shacl/
│   └── eego-shapes.ttl        # SHACL constraint shapes for structural validation
├── kg/
│   ├── chb01_nonsz.trig       # CHB-MIT chb01 non-seizure KG  (9,590 triples, 12 named graphs)
│   ├── chb01_seizure.trig     # CHB-MIT chb01 seizure KG      (11,570 triples, 12 named graphs)
│   ├── S001_rest.trig         # EEGMMIDB S001 rest KG         (38,493 triples, 8 named graphs)
│   └── S001_task.trig         # EEGMMIDB S001 task KG         (116,013 triples, 16 named graphs)
├── src/
│   ├── kg.py                  # RDF namespace definitions and core KG builder
│   ├── kg_pipeline.py         # EDF-to-KG conversion pipeline
│   └── kg_experiments.py      # SPARQL queries (Q1–Q5), FAIR evaluation, SHACL, CQ checks
├── outputs/
│   ├── sparql_queries.json    # Ground-truth SPARQL results (Q1–Q5 for all four KGs)
│   ├── kg_characterisation.json
│   ├── fair_evaluation.json
│   ├── prov_completeness.json
│   ├── shacl_validation.json
│   └── cq_evaluation.json
├── notebooks/
│   └── verify_paper_results.ipynb   # Reproduces all paper tables from the KG files
├── run_full_experiments.py    # Runs the complete pipeline and saves outputs/
└── requirements.txt
```

---

## Datasets

The EDF source files are **not included** in this repository (they are large binary files hosted on PhysioNet). Download them from:

- **CHB-MIT Scalp EEG Database** — https://physionet.org/content/chbmit/
- **EEG Motor Movement/Imagery Dataset (EEGMMIDB)** — https://physionet.org/content/eegmmidb/

PhysioNet DOI: [10.13026/C28G6P](https://doi.org/10.13026/C28G6P)

The four pre-generated KG files in `kg/` allow all paper results to be reproduced **without downloading any EDF files**.

---

## Permanent data deposit

The full dataset (175,666 triples across all four KGs) is permanently archived on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23574953.svg)](https://doi.org/10.5281/zenodo.23574953)

A live SPARQL 1.1 endpoint is available at:  
https://api.triplydb.com/datasets/AlbertoNogalesMoyano/Knowledge-Graph-EEG/sparql

The eego: ontology is registered on BioPortal:  
https://bioportal.bioontology.org/ontologies/EEGO

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.10.

---

## Reproducing paper results

### Option 1 — Notebook (recommended for reviewers)

```bash
jupyter notebook notebooks/verify_paper_results.ipynb
```

The notebook loads the four KG files from `kg/`, runs all SPARQL queries (Q1–Q5), SHACL validation, PROV-O completeness checks, FAIR evaluation, and prints a comparison table against the values reported in the paper. No EDF files required.

### Option 2 — Full pipeline from EDF

Place the EDF files as described in `run_full_experiments.py`, then:

```bash
python run_full_experiments.py
```

This regenerates all four KGs in `outputs/kg/` and all JSON outputs in `outputs/`.

---

## Ontology

The eego: application ontology (`ontology/eego.ttl`) defines:

- **19 OWL classes** — core sensor, connectivity, provenance and annotation classes
- **29 properties** — 18 object properties + 11 datatype properties
- **384 named individuals** — 343 standard electrode-position individuals (10/20, 10/10, 10/5 systems) + 41 connectivity-measure, frequency-band, recording-condition and threshold-rule individuals

Namespace: `http://eegraph.org/ontology#`

---

## Licence

Code: MIT licence  
Knowledge graphs and ontology: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## Citation

If you use this code or the knowledge graphs, please cite:

```bibtex
@article{nogales2026eegkg,
  title   = {Representing {EEG} Functional Connectivity as a {FAIR} Knowledge Graph},
  author  = {Nogales, Alberto and Garc{\'i}a-Tejedor, {\'A}lvaro J.},
  journal = {--},
  year    = {2026},
  doi     = {--}
}
```
