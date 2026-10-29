"""
EEG connectivity → Knowledge Graph exporter.

Converts windowed connectivity data to an RDF/Turtle knowledge graph using
established Semantic Web ontologies (SOSA, OWL-Time, PROV-O, SKOS, SIO,
Schema.org) and the eego: application ontology for electrode position identity.

Supports both monopolar (eego:Electrode / sosa:Sensor nodes) and bipolar
montages (eego:BipolarChannel nodes with eego:positivePole / eego:negativePole).

Designed for use with EEGraph output but accepts any windowed connectivity data
in the form of a dict mapping integer window indices to NetworkX weighted graphs
whose nodes are channel label strings and whose edges carry a ``weight``
attribute. See ``EEGKGExporter`` for the generic class and ``EEGraphKGExporter``
for the EEGraph-specific subclass with default provenance values.
"""

from __future__ import annotations

import datetime
import re
from typing import Dict, List, Optional

import numpy as np

import networkx as nx
from rdflib import (
    BNode, ConjunctiveGraph, Graph as RDFGraph,
    Literal, Namespace, URIRef
)
from rdflib.namespace import OWL, PROV, RDF, RDFS, SKOS, XSD

# ── Established namespaces ────────────────────────────────────────────────────
SOSA   = Namespace("http://www.w3.org/ns/sosa/")
TIME   = Namespace("http://www.w3.org/2006/time#")
SCHEMA = Namespace("https://schema.org/")
EEGO   = Namespace("http://eegraph.org/ontology#")
EEGR   = Namespace("http://eegraph.org/resource/")

# ── Measure metadata ──────────────────────────────────────────────────────────

_SINGLE_CHANNEL = {"power_spectrum", "spectral_entropy", "shannon_entropy"}
_DIRECTED       = {"dtf", "granger_causality", "transfer_entropy", "pdc"}

_DEFAULTS: Dict[str, float] = {
    "pearson_correlation": 0.7, "cross_correlation": 0.5,
    "corr_cross_correlation": 0.1, "squared_coherence": 0.65,
    "imag_coherence": 0.4, "wpli": 0.45, "plv": 0.8,
    "pli": 0.1, "pli_bands": 0.1, "dtf": 0.3,
    "granger_causality": 0.1, "transfer_entropy": 0.05, "pdc": 0.3,
    "power_spectrum": 0.25, "spectral_entropy": 0.25, "shannon_entropy": 0.25,
}

_RULE: Dict[str, str] = {
    "pearson_correlation": "ValueAbove", "squared_coherence": "ValueAbove",
    "wpli": "ValueAbove", "plv": "ValueAbove", "pli": "ValueAbove",
    "pli_bands": "ValueAbove", "dtf": "ValueAbove",
    "granger_causality": "ValueAbove", "transfer_entropy": "ValueAbove",
    "pdc": "ValueAbove",
    "cross_correlation": "MagnitudeAboveUndirected",
    "corr_cross_correlation": "MagnitudeAboveUndirected",
    "imag_coherence": "MagnitudeAboveUndirected",
    "power_spectrum": "TopFraction", "spectral_entropy": "TopFraction",
    "shannon_entropy": "TopFraction",
}

# Map EEGraph measure name → eego: ontology IRI local name
_MEASURE_IRI: Dict[str, str] = {
    "pearson_correlation":    "PearsonCorrelation",
    "cross_correlation":      "CrossCorrelation",
    "corr_cross_correlation": "CrossCorrelationAsymmetry",
    "squared_coherence":      "SquaredCoherence",
    "imag_coherence":         "ImaginaryCoherence",
    "wpli":                   "WPLI",
    "plv":                    "PLV",
    "pli":                    "PLI",
    "pli_bands":              "PLI",
    "dtf":                    "DTF",
    "granger_causality":      "GrangerCausality",
    "transfer_entropy":       "TransferEntropy",
    "pdc":                    "PDC",
    "power_spectrum":         "PowerSpectrum",
    "spectral_entropy":       "SpectralEntropy",
    "shannon_entropy":        "ShannonEntropy",
}

# Map band name → eego: frequency band IRI local name
_BAND_IRI: Dict[str, str] = {
    "delta": "Delta", "theta": "Theta", "alpha": "Alpha",
    "beta": "Beta", "gamma": "Gamma",
}

# Electrode positions defined in the eego ontology (eego.ttl).
# Covers all 343 positions of the 10/20, 10/10, and 10/5 systems
# (Jurcak et al. 2007, via MNE-Python standard_1005 montage).
_EEGO_ELECTRODES: set[str] = {
    "A1",
    "A2",
    "AF1",
    "AF10",
    "AF10H",
    "AF1H",
    "AF2",
    "AF2H",
    "AF3",
    "AF3H",
    "AF4",
    "AF4H",
    "AF5",
    "AF5H",
    "AF6",
    "AF6H",
    "AF7",
    "AF7H",
    "AF8",
    "AF8H",
    "AF9",
    "AF9H",
    "AFF1",
    "AFF10",
    "AFF10H",
    "AFF1H",
    "AFF2",
    "AFF2H",
    "AFF3",
    "AFF3H",
    "AFF4",
    "AFF4H",
    "AFF5",
    "AFF5H",
    "AFF6",
    "AFF6H",
    "AFF7",
    "AFF7H",
    "AFF8",
    "AFF8H",
    "AFF9",
    "AFF9H",
    "AFFZ",
    "AFP1",
    "AFP10",
    "AFP10H",
    "AFP1H",
    "AFP2",
    "AFP2H",
    "AFP3",
    "AFP3H",
    "AFP4",
    "AFP4H",
    "AFP5",
    "AFP5H",
    "AFP6",
    "AFP6H",
    "AFP7",
    "AFP7H",
    "AFP8",
    "AFP8H",
    "AFP9",
    "AFP9H",
    "AFPZ",
    "AFZ",
    "C1",
    "C1H",
    "C2",
    "C2H",
    "C3",
    "C3H",
    "C4",
    "C4H",
    "C5",
    "C5H",
    "C6",
    "C6H",
    "CCP1",
    "CCP1H",
    "CCP2",
    "CCP2H",
    "CCP3",
    "CCP3H",
    "CCP4",
    "CCP4H",
    "CCP5",
    "CCP5H",
    "CCP6",
    "CCP6H",
    "CCPZ",
    "CP1",
    "CP1H",
    "CP2",
    "CP2H",
    "CP3",
    "CP3H",
    "CP4",
    "CP4H",
    "CP5",
    "CP5H",
    "CP6",
    "CP6H",
    "CPP1",
    "CPP1H",
    "CPP2",
    "CPP2H",
    "CPP3",
    "CPP3H",
    "CPP4",
    "CPP4H",
    "CPP5",
    "CPP5H",
    "CPP6",
    "CPP6H",
    "CPPZ",
    "CPZ",
    "CZ",
    "F1",
    "F10",
    "F10H",
    "F1H",
    "F2",
    "F2H",
    "F3",
    "F3H",
    "F4",
    "F4H",
    "F5",
    "F5H",
    "F6",
    "F6H",
    "F7",
    "F7H",
    "F8",
    "F8H",
    "F9",
    "F9H",
    "FC1",
    "FC1H",
    "FC2",
    "FC2H",
    "FC3",
    "FC3H",
    "FC4",
    "FC4H",
    "FC5",
    "FC5H",
    "FC6",
    "FC6H",
    "FCC1",
    "FCC1H",
    "FCC2",
    "FCC2H",
    "FCC3",
    "FCC3H",
    "FCC4",
    "FCC4H",
    "FCC5",
    "FCC5H",
    "FCC6",
    "FCC6H",
    "FCCZ",
    "FCZ",
    "FFC1",
    "FFC1H",
    "FFC2",
    "FFC2H",
    "FFC3",
    "FFC3H",
    "FFC4",
    "FFC4H",
    "FFC5",
    "FFC5H",
    "FFC6",
    "FFC6H",
    "FFCZ",
    "FFT10",
    "FFT10H",
    "FFT7",
    "FFT7H",
    "FFT8",
    "FFT8H",
    "FFT9",
    "FFT9H",
    "FP1",
    "FP1H",
    "FP2",
    "FP2H",
    "FPZ",
    "FT10",
    "FT10H",
    "FT7",
    "FT7H",
    "FT8",
    "FT8H",
    "FT9",
    "FT9H",
    "FTT10",
    "FTT10H",
    "FTT7",
    "FTT7H",
    "FTT8",
    "FTT8H",
    "FTT9",
    "FTT9H",
    "FZ",
    "I1",
    "I1H",
    "I2",
    "I2H",
    "IZ",
    "M1",
    "M2",
    "O1",
    "O1H",
    "O2",
    "O2H",
    "OI1",
    "OI1H",
    "OI2",
    "OI2H",
    "OIZ",
    "OZ",
    "P1",
    "P10",
    "P10H",
    "P1H",
    "P2",
    "P2H",
    "P3",
    "P3H",
    "P4",
    "P4H",
    "P5",
    "P5H",
    "P6",
    "P6H",
    "P7",
    "P7H",
    "P8",
    "P8H",
    "P9",
    "P9H",
    "PO1",
    "PO10",
    "PO10H",
    "PO1H",
    "PO2",
    "PO2H",
    "PO3",
    "PO3H",
    "PO4",
    "PO4H",
    "PO5",
    "PO5H",
    "PO6",
    "PO6H",
    "PO7",
    "PO7H",
    "PO8",
    "PO8H",
    "PO9",
    "PO9H",
    "POO1",
    "POO10",
    "POO10H",
    "POO1H",
    "POO2",
    "POO2H",
    "POO3",
    "POO3H",
    "POO4",
    "POO4H",
    "POO5",
    "POO5H",
    "POO6",
    "POO6H",
    "POO7",
    "POO7H",
    "POO8",
    "POO8H",
    "POO9",
    "POO9H",
    "POOZ",
    "POZ",
    "PPO1",
    "PPO10",
    "PPO10H",
    "PPO1H",
    "PPO2",
    "PPO2H",
    "PPO3",
    "PPO3H",
    "PPO4",
    "PPO4H",
    "PPO5",
    "PPO5H",
    "PPO6",
    "PPO6H",
    "PPO7",
    "PPO7H",
    "PPO8",
    "PPO8H",
    "PPO9",
    "PPO9H",
    "PPOZ",
    "PZ",
    "T10",
    "T10H",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "T7H",
    "T8",
    "T8H",
    "T9",
    "T9H",
    "TP10",
    "TP10H",
    "TP7",
    "TP7H",
    "TP8",
    "TP8H",
    "TP9",
    "TP9H",
    "TPP10",
    "TPP10H",
    "TPP7",
    "TPP7H",
    "TPP8",
    "TPP8H",
    "TPP9",
    "TPP9H",
    "TTP10",
    "TTP10H",
    "TTP7",
    "TTP7H",
    "TTP8",
    "TTP8H",
    "TTP9",
    "TTP9H"
}

# 2-D scalp coordinates: azimuthal equidistant projection centred at Cz,
# scaled so T7/T8 are at x=±0.80 (Jurcak et al. 2007 via MNE standard_1005).
_SCALP_POS = {
    "A1": (-0.9826, -0.1331),
    "A2": (0.9893, -0.1355),
    "AF1": (-0.1165, 0.5553),
    "AF10": (0.4839, 0.742),
    "AF10H": (0.4687, 0.6722),
    "AF1H": (-0.0587, 0.5535),
    "AF2": (0.1189, 0.5556),
    "AF2H": (0.061, 0.5528),
    "AF3": (-0.2245, 0.5753),
    "AF3H": (-0.174, 0.5604),
    "AF4": (0.2283, 0.5718),
    "AF4H": (0.1759, 0.5605),
    "AF5": (-0.327, 0.5986),
    "AF5H": (-0.2752, 0.5887),
    "AF6": (0.3258, 0.6),
    "AF6H": (0.2752, 0.5889),
    "AF7": (-0.4249, 0.6154),
    "AF7H": (-0.3796, 0.6038),
    "AF8": (0.4215, 0.6193),
    "AF8H": (0.3766, 0.6074),
    "AF9": (-0.4787, 0.742),
    "AF9H": (-0.4715, 0.6688),
    "AFF1": (-0.1435, 0.478),
    "AFF10": (0.6104, 0.6616),
    "AFF10H": (0.5754, 0.6035),
    "AFF1H": (-0.0702, 0.4777),
    "AFF2": (0.1485, 0.476),
    "AFF2H": (0.0763, 0.4745),
    "AFF3": (-0.2816, 0.494),
    "AFF3H": (-0.2138, 0.4839),
    "AFF4": (0.2846, 0.495),
    "AFF4H": (0.218, 0.484),
    "AFF5": (-0.4061, 0.5239),
    "AFF5H": (-0.3464, 0.506),
    "AFF6": (0.4084, 0.5235),
    "AFF6H": (0.3479, 0.5082),
    "AFF7": (-0.5201, 0.5536),
    "AFF7H": (-0.4687, 0.5328),
    "AFF8": (0.518, 0.5577),
    "AFF8H": (0.4681, 0.5359),
    "AFF9": (-0.6049, 0.6617),
    "AFF9H": (-0.5773, 0.5995),
    "AFFZ": (-0.0009, 0.4748),
    "AFP1": (-0.0837, 0.6384),
    "AFP10": (0.3514, 0.8004),
    "AFP10H": (0.3552, 0.7251),
    "AFP1H": (-0.0415, 0.6343),
    "AFP2": (0.0857, 0.6339),
    "AFP2H": (0.0431, 0.6332),
    "AFP3": (-0.1588, 0.6583),
    "AFP3H": (-0.1199, 0.65),
    "AFP4": (0.1618, 0.6523),
    "AFP4H": (0.1233, 0.6438),
    "AFP5": (-0.2423, 0.6631),
    "AFP5H": (-0.201, 0.6609),
    "AFP6": (0.239, 0.6645),
    "AFP6H": (0.1982, 0.6618),
    "AFP7": (-0.3237, 0.6616),
    "AFP7H": (-0.2835, 0.6626),
    "AFP8": (0.319, 0.665),
    "AFP8H": (0.2804, 0.6635),
    "AFP9": (-0.3454, 0.8003),
    "AFP9H": (-0.3601, 0.7216),
    "AFPZ": (-0.0018, 0.6306),
    "AFZ": (-0.0013, 0.5552),
    "C1": (-0.2039, -0.0022),
    "C1H": (-0.1001, -0.0007),
    "C2": (0.2109, 0.0004),
    "C2H": (0.1054, 0.0008),
    "C3": (-0.412, -0.0078),
    "C3H": (-0.3083, -0.0044),
    "C4": (0.4193, -0.0031),
    "C4H": (0.3156, -0.0008),
    "C5": (-0.613, -0.0171),
    "C5H": (-0.5127, -0.012),
    "C6": (0.6163, -0.0103),
    "C6H": (0.5181, -0.0064),
    "CCP1": (-0.2016, -0.1037),
    "CCP1H": (-0.099, -0.101),
    "CCP2": (0.2076, -0.1024),
    "CCP2H": (0.1046, -0.1001),
    "CCP3": (-0.4045, -0.1161),
    "CCP3H": (-0.3036, -0.1091),
    "CCP4": (0.4096, -0.1111),
    "CCP4H": (0.3094, -0.1057),
    "CCP5": (-0.6011, -0.1382),
    "CCP5H": (-0.5036, -0.1249),
    "CCP6": (0.6031, -0.1304),
    "CCP6H": (0.5074, -0.1189),
    "CCPZ": (-0.0, -0.0991),
    "CP1": (-0.1881, -0.1999),
    "CP1H": (-0.091, -0.1968),
    "CP2": (0.1994, -0.1991),
    "CP2H": (0.1029, -0.1946),
    "CP3": (-0.3807, -0.2199),
    "CP3H": (-0.2849, -0.2083),
    "CP4": (0.3897, -0.2159),
    "CP4H": (0.2942, -0.2054),
    "CP5": (-0.5691, -0.2517),
    "CP5H": (-0.4762, -0.2338),
    "CP6": (0.5749, -0.2435),
    "CP6H": (0.4839, -0.2283),
    "CPP1": (-0.1733, -0.2941),
    "CPP1H": (-0.0834, -0.2912),
    "CPP2": (0.1847, -0.2918),
    "CPP2H": (0.0963, -0.287),
    "CPP3": (-0.3512, -0.3167),
    "CPP3H": (-0.262, -0.3053),
    "CPP4": (0.3598, -0.3126),
    "CPP4H": (0.2731, -0.2993),
    "CPP5": (-0.5216, -0.3577),
    "CPP5H": (-0.4375, -0.3348),
    "CPP6": (0.5286, -0.3506),
    "CPP6H": (0.4456, -0.3292),
    "CPPZ": (-0.0001, -0.2872),
    "CPZ": (-0.0, -0.1955),
    "CZ": (0.0, 0.0),
    "F1": (-0.164, 0.3929),
    "F10": (0.7197, 0.5517),
    "F10H": (0.6761, 0.5082),
    "F1H": (-0.08, 0.3925),
    "F2": (0.1709, 0.3969),
    "F2H": (0.0882, 0.3905),
    "F3": (-0.3261, 0.4095),
    "F3H": (-0.2464, 0.3991),
    "F4": (0.3308, 0.4175),
    "F4H": (0.2522, 0.4051),
    "F5": (-0.4771, 0.4361),
    "F5H": (-0.4034, 0.4215),
    "F6": (0.4824, 0.4364),
    "F6H": (0.4072, 0.4305),
    "F7": (-0.6104, 0.4711),
    "F7H": (-0.5468, 0.4515),
    "F8": (0.6095, 0.4742),
    "F8H": (0.5494, 0.4517),
    "F9": (-0.7164, 0.5526),
    "F9H": (-0.6768, 0.504),
    "FC1": (-0.1967, 0.204),
    "FC1H": (-0.0977, 0.201),
    "FC2": (0.1987, 0.2099),
    "FC2H": (0.0996, 0.2036),
    "FC3": (-0.3938, 0.2164),
    "FC3H": (-0.2958, 0.2095),
    "FC4": (0.397, 0.22),
    "FC4H": (0.2985, 0.2153),
    "FC5": (-0.581, 0.2248),
    "FC5H": (-0.4897, 0.2168),
    "FC6": (0.5818, 0.2302),
    "FC6H": (0.4915, 0.222),
    "FCC1": (-0.2058, 0.1027),
    "FCC1H": (-0.1025, 0.1023),
    "FCC2": (0.2076, 0.108),
    "FCC2H": (0.1024, 0.105),
    "FCC3": (-0.4112, 0.1038),
    "FCC3H": (-0.3093, 0.1038),
    "FCC4": (0.4157, 0.1102),
    "FCC4H": (0.3126, 0.1093),
    "FCC5": (-0.6065, 0.1056),
    "FCC5H": (-0.51, 0.103),
    "FCC6": (0.6082, 0.1133),
    "FCC6H": (0.513, 0.1082),
    "FCCZ": (-0.0001, 0.1018),
    "FCZ": (-0.0002, 0.2019),
    "FFC1": (-0.1804, 0.3035),
    "FFC1H": (-0.0884, 0.2973),
    "FFC2": (0.1862, 0.306),
    "FFC2H": (0.0954, 0.2978),
    "FFC3": (-0.3637, 0.3178),
    "FFC3H": (-0.2728, 0.3099),
    "FFC4": (0.3679, 0.3251),
    "FFC4H": (0.2773, 0.3166),
    "FFC5": (-0.5359, 0.3374),
    "FFC5H": (-0.4521, 0.326),
    "FFC6": (0.5391, 0.342),
    "FFC6H": (0.4558, 0.3331),
    "FFCZ": (-0.0004, 0.298),
    "FFT10": (0.8107, 0.4269),
    "FFT10H": (0.7601, 0.3927),
    "FFT7": (-0.6897, 0.3662),
    "FFT7H": (-0.6147, 0.354),
    "FFT8": (0.6879, 0.3728),
    "FFT8H": (0.6161, 0.3552),
    "FFT9": (-0.8076, 0.4256),
    "FFT9H": (-0.7593, 0.388),
    "FP1": (-0.2177, 0.6929),
    "FP1H": (-0.1098, 0.7073),
    "FP2": (0.2125, 0.6949),
    "FP2H": (0.1048, 0.708),
    "FPZ": (-0.0026, 0.7052),
    "FT10": (0.8809, 0.2883),
    "FT10H": (0.8259, 0.2624),
    "FT7": (-0.7503, 0.2452),
    "FT7H": (-0.668, 0.2348),
    "FT8": (0.7493, 0.2568),
    "FT8H": (0.6681, 0.2439),
    "FT9": (-0.8759, 0.2845),
    "FT9H": (-0.8234, 0.2569),
    "FTT10": (0.928, 0.1331),
    "FTT10H": (0.8688, 0.1156),
    "FTT7": (-0.7885, 0.1101),
    "FTT7H": (-0.7003, 0.1088),
    "FTT8": (0.7904, 0.1222),
    "FTT8H": (0.7014, 0.1175),
    "FTT9": (-0.9186, 0.13),
    "FTT9H": (-0.8641, 0.1124),
    "FZ": (-0.0006, 0.3897),
    "I1": (-0.2507, -0.8485),
    "I1H": (-0.1275, -0.866),
    "I2": (0.2438, -0.8528),
    "I2H": (0.1195, -0.8682),
    "IZ": (-0.004, -0.8672),
    "M1": (-0.9295, -0.3427),
    "M2": (0.9341, -0.3493),
    "O1": (-0.2107, -0.7146),
    "O1H": (-0.1067, -0.7267),
    "O2": (0.2084, -0.7167),
    "O2H": (0.1026, -0.7278),
    "OI1": (-0.2278, -0.7852),
    "OI1H": (-0.115, -0.799),
    "OI2": (0.2246, -0.7886),
    "OI2H": (0.1097, -0.8004),
    "OIZ": (-0.0032, -0.7963),
    "OZ": (-0.0025, -0.7238),
    "P1": (-0.1577, -0.3864),
    "P10": (0.7106, -0.598),
    "P10H": (0.6531, -0.5632),
    "P1H": (-0.0759, -0.3795),
    "P2": (0.1682, -0.3801),
    "P2H": (0.0885, -0.3763),
    "P3": (-0.3184, -0.4095),
    "P3H": (-0.2387, -0.3972),
    "P4": (0.3254, -0.404),
    "P4H": (0.2476, -0.391),
    "P5": (-0.467, -0.4504),
    "P5H": (-0.3948, -0.4264),
    "P6": (0.4698, -0.4521),
    "P6H": (0.4002, -0.4235),
    "P7": (-0.5996, -0.506),
    "P7H": (-0.5344, -0.4797),
    "P8": (0.6027, -0.5076),
    "P8H": (0.5404, -0.4756),
    "P9": (-0.7118, -0.5916),
    "P9H": (-0.653, -0.5596),
    "PO1": (-0.1197, -0.5641),
    "PO10": (0.4877, -0.768),
    "PO10H": (0.4471, -0.7112),
    "PO1H": (-0.0606, -0.5598),
    "PO2": (0.1201, -0.5653),
    "PO2H": (0.06, -0.5621),
    "PO3": (-0.2348, -0.5742),
    "PO3H": (-0.178, -0.5692),
    "PO4": (0.2332, -0.5796),
    "PO4H": (0.1771, -0.5714),
    "PO5": (-0.3329, -0.6023),
    "PO5H": (-0.2866, -0.5857),
    "PO6": (0.3354, -0.6019),
    "PO6H": (0.2873, -0.5878),
    "PO7": (-0.4141, -0.6444),
    "PO7H": (-0.3744, -0.6235),
    "PO8": (0.4142, -0.6468),
    "PO8H": (0.3749, -0.6254),
    "PO9": (-0.4919, -0.7613),
    "PO9H": (-0.4496, -0.7086),
    "POO1": (-0.0911, -0.6371),
    "POO10": (0.367, -0.8199),
    "POO10H": (0.337, -0.7571),
    "POO1H": (-0.048, -0.6424),
    "POO2": (0.0867, -0.6466),
    "POO2H": (0.0418, -0.648),
    "POO3": (-0.1749, -0.6476),
    "POO3H": (-0.133, -0.6434),
    "POO4": (0.1738, -0.6493),
    "POO4H": (0.1309, -0.6481),
    "POO5": (-0.2508, -0.6614),
    "POO5H": (-0.2162, -0.6504),
    "POO6": (0.2535, -0.66),
    "POO6H": (0.2135, -0.6559),
    "POO7": (-0.3136, -0.6874),
    "POO7H": (-0.2843, -0.6728),
    "POO8": (0.3124, -0.6897),
    "POO8H": (0.2846, -0.6737),
    "POO9": (-0.3724, -0.8145),
    "POO9H": (-0.3397, -0.7554),
    "POOZ": (-0.0019, -0.6439),
    "POZ": (-0.0013, -0.5575),
    "PPO1": (-0.1442, -0.4742),
    "PPO10": (0.604, -0.6948),
    "PPO10H": (0.553, -0.6489),
    "PPO1H": (-0.071, -0.4717),
    "PPO2": (0.1479, -0.4694),
    "PPO2H": (0.0757, -0.4658),
    "PPO3": (-0.2841, -0.4919),
    "PPO3H": (-0.2148, -0.483),
    "PPO4": (0.2851, -0.4913),
    "PPO4H": (0.2183, -0.4773),
    "PPO5": (-0.4048, -0.532),
    "PPO5H": (-0.3477, -0.5068),
    "PPO6": (0.4101, -0.5274),
    "PPO6H": (0.3487, -0.5092),
    "PPO7": (-0.5102, -0.5845),
    "PPO7H": (-0.4614, -0.5545),
    "PPO8": (0.5115, -0.5872),
    "PPO8H": (0.461, -0.5585),
    "PPO9": (-0.6061, -0.6886),
    "PPO9H": (-0.5547, -0.6439),
    "PPOZ": (-0.0008, -0.4681),
    "PZ": (-0.0004, -0.3777),
    "T10": (0.9435, -0.0353),
    "T10H": (0.8781, -0.0432),
    "T3": (-0.7985, -0.0328),
    "T4": (0.8026, -0.0233),
    "T5": (-0.5996, -0.506),
    "T6": (0.6027, -0.5076),
    "T7": (-0.7985, -0.0328),
    "T7H": (-0.7097, -0.0239),
    "T8": (0.8026, -0.0233),
    "T8H": (0.7124, -0.0158),
    "T9": (-0.9365, -0.0297),
    "T9H": (-0.8743, -0.0428),
    "TP10": (0.8703, -0.3475),
    "TP10H": (0.8089, -0.33),
    "TP7": (-0.7388, -0.2932),
    "TP7H": (-0.6577, -0.2734),
    "TP8": (0.7436, -0.2916),
    "TP8H": (0.6618, -0.2668),
    "TP9": (-0.8682, -0.3397),
    "TP9H": (-0.8056, -0.3276),
    "TPP10": (0.8008, -0.4791),
    "TPP10H": (0.7414, -0.4578),
    "TPP7": (-0.6781, -0.4081),
    "TPP7H": (-0.6023, -0.3838),
    "TPP8": (0.6825, -0.4083),
    "TPP8H": (0.6073, -0.3798),
    "TPP9": (-0.8005, -0.4722),
    "TPP9H": (-0.74, -0.4536),
    "TTP10": (0.9209, -0.1997),
    "TTP10H": (0.8543, -0.1902),
    "TTP7": (-0.7798, -0.1691),
    "TTP7H": (-0.6933, -0.1516),
    "TTP8": (0.7855, -0.1641),
    "TTP8H": (0.6971, -0.1464),
    "TTP9": (-0.9164, -0.192),
    "TTP9H": (-0.8515, -0.1902)
}


def _eego_electrode_uri(label: str) -> Optional[URIRef]:
    """Return the eego ontology URI for a known electrode label (case-insensitive)."""
    key = label.upper()
    if key in _EEGO_ELECTRODES:
        return EEGO[f"Electrode_{key}"]
    return None


def _scalp_pos(label: str) -> Optional[tuple]:
    """Return (x, y) scalp position for an electrode label (case-insensitive)."""
    return _SCALP_POS.get(label.upper())


def _looks_bipolar(ch: str) -> bool:
    """Heuristic: True if ch looks like a bipolar differential channel (e.g. 'FP1-F7')."""
    return bool(re.match(r"^[A-Za-z0-9]+-[A-Za-z0-9]+$", ch))


def _bipolar_poles(ch: str) -> tuple[str, str]:
    """Split 'FP1-F7' → ('FP1', 'F7').

    MNE appends a numeric suffix to duplicate derivations (e.g. 'T8-P8-0',
    'T8-P8-1'); the suffix is part of the channel IRI but must be stripped
    when extracting the negative pole electrode label.
    """
    parts = ch.split("-")
    pos = parts[0].upper()
    # Strip a trailing all-digit token (MNE duplicate suffix) from the neg pole
    neg_raw = parts[1]
    neg = neg_raw.upper()
    return pos, neg


class EEGKGExporter:
    """Convert windowed EEG connectivity data to an RDF Knowledge Graph.

    Parameters
    ----------
    graphs_dict : dict[int, nx.Graph]
        Windowed connectivity data: integer window index → NetworkX undirected
        weighted graph. Each graph's nodes must be channel label strings and
        each edge must carry a ``weight`` attribute (float).
    ch_names : list[str]
        Ordered list of channel / electrode labels. Must cover all nodes
        present in any graph in ``graphs_dict``.
    connectivity : str
        Name of the connectivity measure (e.g. ``'pearson_correlation'``).
    threshold : float, optional
        Edge-weight threshold applied upstream. Stored as provenance metadata.
    bands : list[str], optional
        Frequency-band names if the windows are band-stratified
        (e.g. ``['delta', 'alpha']``). Pass ``None`` for broadband measures.
    eeg_file : str, optional
        Source EEG filename for provenance (e.g. ``'chb01_03.edf'``).
    window_size : int, optional
        Time-window duration in seconds.
    montage : str, optional
        ``'monopolar'``, ``'bipolar'``, or ``'auto'`` (default).
        ``'auto'`` infers the type from channel names: any name matching the
        pattern ``LABEL-LABEL`` (e.g. ``FP1-F7``) is treated as bipolar.
        Pass ``'monopolar'`` or ``'bipolar'`` explicitly when channel names
        do not follow this convention.
    tool_name : str, optional
        Name of the software tool that produced the connectivity data.
    tool_version : str, optional
        Version string of the tool.
    tool_url : str, optional
        Repository or homepage URL of the tool.
    """

    def __init__(
        self,
        graphs_dict: Dict[int, nx.Graph],
        ch_names: List[str],
        connectivity: str = "unknown",
        threshold: Optional[float] = None,
        bands: Optional[List] = None,
        eeg_file: str = "unknown.edf",
        window_size: int = 5,
        montage: str = "auto",
        tool_name: str = "unknown",
        tool_version: str = "unknown",
        tool_url: str = "https://example.org",
        recording_condition: Optional[str] = None,
        annotations: Optional[List[Dict]] = None,
        sample_rate: Optional[float] = None,
        segment_start: Optional[float] = None,
        segment_end: Optional[float] = None,
        dataset_url: Optional[str] = None,
    ):
        if montage not in ("auto", "monopolar", "bipolar"):
            raise ValueError("montage must be 'auto', 'monopolar', or 'bipolar'")

        self.graphs_dict = graphs_dict
        self.ch_names = ch_names
        self.connectivity = connectivity
        self.bands = bands or []
        self.eeg_file = eeg_file
        self.window_size = window_size
        self.tool_name = tool_name
        self.tool_version = tool_version
        self.tool_url = tool_url
        self.recording_condition = recording_condition
        self.annotations: List[Dict] = annotations or []
        self.sample_rate: Optional[float] = sample_rate
        self.segment_start: Optional[float] = segment_start
        self.segment_end: Optional[float] = segment_end
        self.dataset_url: Optional[str] = dataset_url
        self._cg: Optional[ConjunctiveGraph] = None

        # Effective threshold: caller value if supplied, else measure default
        self.threshold: float = (
            threshold if threshold is not None
            else _DEFAULTS.get(connectivity, 0.3)
        )
        self._threshold_rule: Optional[str] = _RULE.get(connectivity)
        self._is_directed = connectivity in _DIRECTED
        self._is_single_channel = connectivity in _SINGLE_CHANNEL
        # eego: measure IRI (None for unknown measures)
        self._measure_uri: Optional[URIRef] = (
            EEGO[_MEASURE_IRI[connectivity]] if connectivity in _MEASURE_IRI else None
        )

        if montage == "auto":
            self._is_bipolar_montage = any(_looks_bipolar(ch) for ch in ch_names)
        else:
            self._is_bipolar_montage = (montage == "bipolar")

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(self) -> ConjunctiveGraph:
        """Validate inputs and build the ConjunctiveGraph."""
        self._validate()
        cg = ConjunctiveGraph()
        for prefix, ns in [
            ("sosa", SOSA), ("time", TIME), ("schema", SCHEMA),
            ("skos", SKOS),
            ("prov", PROV), ("owl", OWL), ("eego", EEGO), ("eegr", EEGR),
        ]:
            cg.bind(prefix, ns)

        self._add_ontology_import(cg)
        self._add_software_agent(cg)
        self._add_electrodes(cg)
        self._add_recording(cg)
        self._add_windows(cg)
        if self.annotations:
            self._add_annotated_events(cg)
        self._cg = cg
        return cg

    def validate_shacl(self, shapes_path: str | None = None) -> tuple[bool, str]:
        """Validate the built KG against the eego SHACL shapes graph.

        Returns (conforms, report_text). Raises RuntimeError if build() was not
        called first. If shapes_path is None, looks for eego-shapes.ttl next to
        this file.
        """
        try:
            from pyshacl import validate as shacl_validate
        except ImportError as exc:
            raise ImportError("pyshacl is required for SHACL validation: pip install pyshacl") from exc

        if self._cg is None:
            raise RuntimeError("Call build() before validate_shacl().")

        if shapes_path is None:
            import os
            shapes_path = os.path.join(os.path.dirname(__file__), "eego-shapes.ttl")

        from rdflib import Graph as RDFGraph
        shapes_graph = RDFGraph()
        shapes_graph.parse(shapes_path, format="turtle")

        # Load the eego ontology as ont_graph so pyshacl can resolve named
        # individual types (e.g. eego:PearsonCorrelation a eego:ConnectivityMeasure)
        # which live in eego.ttl rather than in the KG itself.
        ont_path = os.path.join(os.path.dirname(__file__), "eego.ttl")
        ont_graph = RDFGraph()
        ont_graph.parse(ont_path, format="turtle")

        conforms, _, report_text = shacl_validate(
            self._cg,
            shacl_graph=shapes_graph,
            ont_graph=ont_graph,
            inference="rdfs",
            abort_on_first=False,
        )
        return conforms, report_text

    def to_turtle(self, path: str) -> None:
        """Serialise the KG to a TriG file (named-graph-aware superset of Turtle)."""
        if self._cg is None:
            self.build()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self._cg.serialize(format="trig"))

    def to_string(self, fmt: str = "trig") -> str:
        """Serialise the KG to a string."""
        if self._cg is None:
            self.build()
        return self._cg.serialize(format=fmt)

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        # Single-channel measures (PSD, entropy) use pole/electrode names as nodes,
        # not the full channel label — skip the ch_names membership check.
        if self._is_single_channel:
            return
        ch_set = set(self.ch_names)
        for win_idx, G in self.graphs_dict.items():
            unknown = set(G.nodes) - ch_set
            if unknown:
                raise ValueError(
                    f"Window {win_idx}: graph nodes {sorted(unknown)} are not "
                    f"in ch_names. Pass the complete channel list to the constructor."
                )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _slug(self, *parts) -> str:
        return "_".join(str(p).replace(" ", "_").replace("/", "-") for p in parts)

    def _add_ontology_import(self, cg: ConjunctiveGraph):
        """Declare owl:imports so the KG is self-describing via the eego ontology IRI."""
        g = cg.get_context(EEGO["ontology"])
        kg_iri = EEGR["kg"]
        eego_iri = URIRef("http://eegraph.org/ontology")
        g.add((kg_iri, RDF.type, OWL.Ontology))
        g.add((kg_iri, OWL.imports, eego_iri))
        g.add((kg_iri, RDFS.label, Literal("EEGraph KG — ontology import declaration")))

    def _add_software_agent(self, cg: ConjunctiveGraph):
        g = cg.get_context(EEGO["provenance"])
        tool_slug = self._slug(self.tool_name)
        sw = EEGR[f"software/{tool_slug}"]
        g.add((sw, RDF.type, SCHEMA.SoftwareApplication))
        g.add((sw, RDF.type, PROV.SoftwareAgent))
        g.add((sw, SCHEMA.name,    Literal(self.tool_name)))
        g.add((sw, SCHEMA.version, Literal(self.tool_version)))
        g.add((sw, SCHEMA.url,     URIRef(self.tool_url)))
        self._sw_uri = sw

    def _add_electrodes(self, cg: ConjunctiveGraph):
        g = cg.get_context(EEGO["electrodes"])

        if self._is_bipolar_montage:
            poles: set[str] = set()
            for ch in self.ch_names:
                if _looks_bipolar(ch):
                    pos, neg = _bipolar_poles(ch)
                    poles.add(pos)
                    poles.add(neg)
                else:
                    poles.add(ch.upper())

            rec_slug = self._slug(self.eeg_file)
            for pole in poles:
                e_uri = EEGR[f"electrode/{rec_slug}/{pole}"]
                g.add((e_uri, RDF.type, EEGO["Electrode"]))
                g.add((e_uri, RDFS.label, Literal(pole)))
                eego_uri = _eego_electrode_uri(pole)
                if eego_uri:
                    g.add((e_uri, EEGO["hasStandardPosition"], eego_uri))
                pos = _scalp_pos(pole)
                if pos:
                    x, y = pos
                    g.add((e_uri, EEGO["scalp2DX"], Literal(x, datatype=XSD.decimal)))
                    g.add((e_uri, EEGO["scalp2DY"], Literal(y, datatype=XSD.decimal)))

            for ch in self.ch_names:
                if _looks_bipolar(ch):
                    pos_pole, neg_pole = _bipolar_poles(ch)
                    ch_uri = EEGR[f"channel/{rec_slug}/{ch}"]
                    g.add((ch_uri, RDF.type, EEGO["BipolarChannel"]))
                    g.add((ch_uri, RDFS.label, Literal(ch)))
                    g.add((ch_uri, EEGO["positivePole"], EEGR[f"electrode/{rec_slug}/{pos_pole}"]))
                    g.add((ch_uri, EEGO["negativePole"], EEGR[f"electrode/{rec_slug}/{neg_pole}"]))
        else:
            rec_slug = self._slug(self.eeg_file)
            for ch in self.ch_names:
                key = ch.upper()
                e_uri = EEGR[f"electrode/{rec_slug}/{ch}"]
                g.add((e_uri, RDF.type, EEGO["Electrode"]))
                g.add((e_uri, RDF.type, SOSA.Sensor))
                g.add((e_uri, RDFS.label, Literal(ch)))
                eego_uri = _eego_electrode_uri(key)
                if eego_uri:
                    g.add((e_uri, EEGO["hasStandardPosition"], eego_uri))
                sp = _scalp_pos(key)
                if sp:
                    x, y = sp
                    g.add((e_uri, EEGO["scalp2DX"], Literal(x, datatype=XSD.decimal)))
                    g.add((e_uri, EEGO["scalp2DY"], Literal(y, datatype=XSD.decimal)))

    def _add_recording(self, cg: ConjunctiveGraph):
        g = cg.get_context(EEGO["provenance"])
        rec = EEGR[f"recording/{self._slug(self.eeg_file)}"]
        g.add((rec, RDF.type, SOSA.ObservationCollection))
        g.add((rec, RDF.type, PROV.Entity))
        g.add((rec, RDFS.label, Literal(self.eeg_file)))
        g.add((rec, PROV.wasAttributedTo, self._sw_uri))
        now = datetime.datetime.utcnow().isoformat()
        g.add((rec, PROV.generatedAtTime, Literal(now, datatype=XSD.dateTime)))

        activity = EEGR[f"activity/{self._slug(self.eeg_file, self.connectivity)}"]
        g.add((activity, RDF.type, PROV.Activity))
        g.add((activity, RDF.type, EEGO["ConnectivityComputation"]))
        g.add((activity, RDFS.label, Literal(f"{self.connectivity} analysis")))
        g.add((activity, PROV.wasAssociatedWith, self._sw_uri))
        g.add((activity, PROV.used, rec))
        if self._measure_uri is not None:
            g.add((activity, EEGO["connectivityMeasure"], self._measure_uri))
        else:
            g.add((activity, EEGO["connectivityMeasure"], Literal(self.connectivity)))
        g.add((activity, EEGO["threshold"], Literal(str(self.threshold), datatype=XSD.decimal)))
        if self._threshold_rule:
            g.add((activity, EEGO["thresholdRule"], EEGO[self._threshold_rule]))
        if self.recording_condition:
            g.add((rec, EEGO["recordingCondition"], EEGO[self.recording_condition]))
        if self.sample_rate is not None:
            g.add((rec, SCHEMA["sampleRate"], Literal(self.sample_rate, datatype=XSD.decimal)))
        if self.segment_start is not None:
            g.add((activity, EEGO["segmentStart"], Literal(self.segment_start, datatype=XSD.decimal)))
        if self.segment_end is not None:
            g.add((activity, EEGO["segmentEnd"], Literal(self.segment_end, datatype=XSD.decimal)))
        if self.dataset_url is not None:
            g.add((rec, PROV.wasDerivedFrom, URIRef(self.dataset_url)))
        self._recording_uri = rec
        self._activity_uri = activity
        self._threshold_rule_uri = EEGO[self._threshold_rule] if self._threshold_rule else None

    def _add_annotated_events(self, cg: ConjunctiveGraph):
        """Emit eego:AnnotatedEvent triples and link overlapping time windows.

        Each annotation dict must have:
            onset      : float  — seconds from recording start
            offset     : float  — seconds from recording start
        Optional keys:
            event_type : str    — either a full URI (e.g. 'http://purl.obolibrary.org/obo/HP_0001250')
                                  or a plain string label stored as eego:eventLabel.
                                  Users supply their own vocabulary (SNOMED, HED, custom SKOS).
            label      : str    — free-text description stored as eego:eventLabel
        """
        g = cg.get_context(EEGO["events"])
        _CANONICAL = ["delta", "theta", "alpha", "beta", "gamma"]
        nb = len([b for b in _CANONICAL
                  if b in [x.lower() if isinstance(x, str) else "" for x in self.bands]])

        for idx, ann in enumerate(self.annotations):
            onset  = float(ann["onset"])
            offset = float(ann["offset"])
            etype  = ann.get("event_type")
            label  = ann.get("label")

            ev_uri = EEGR[f"event/{self._slug(self.eeg_file, idx)}"]
            g.add((ev_uri, RDF.type,           EEGO["AnnotatedEvent"]))
            g.add((ev_uri, EEGO["eventOnset"],  Literal(onset,  datatype=XSD.float)))
            g.add((ev_uri, EEGO["eventOffset"], Literal(offset, datatype=XSD.float)))
            g.add((ev_uri, EEGO["inRecording"], self._recording_uri))

            if etype:
                # Full URI → object property; plain string → eventLabel literal
                if etype.startswith("http://") or etype.startswith("https://"):
                    g.add((ev_uri, EEGO["hasEventType"], URIRef(etype)))
                else:
                    g.add((ev_uri, EEGO["eventLabel"], Literal(etype, datatype=XSD.string)))
            if label:
                g.add((ev_uri, EEGO["eventLabel"], Literal(label, datatype=XSD.string)))

            # Link every time window that temporally overlaps this event.
            # Overlap condition: window_start < event_offset AND window_end > event_onset
            for win_idx in self.graphs_dict:
                window_index = win_idx // nb if nb else win_idx
                win_start = window_index * self.window_size
                win_end   = win_start + self.window_size
                if win_start < offset and win_end > onset:
                    win_uri = EEGR[f"window/{self._slug(self.eeg_file, window_index)}"]
                    ctx = cg.get_context(EEGO[f"window/{self._slug(self.eeg_file, win_idx)}"])
                    ctx.add((win_uri, TIME["intervalOverlaps"], ev_uri))

    def _add_windows(self, cg: ConjunctiveGraph):
        now = datetime.datetime.utcnow().isoformat()
        _CANONICAL = ["delta", "theta", "alpha", "beta", "gamma"]
        active_bands = [b for b in _CANONICAL
                        if b in [x.lower() if isinstance(x, str) else "" for x in self.bands]]
        nb = len(active_bands)

        for win_idx, G in self.graphs_dict.items():
            if nb:
                window_index = win_idx // nb
                band_name = active_bands[win_idx % nb]
            else:
                window_index = win_idx
                band_name = None

            win_uri  = EEGR[f"window/{self._slug(self.eeg_file, window_index)}"]
            ctx_name = EEGO[f"window/{self._slug(self.eeg_file, win_idx)}"]
            g = cg.get_context(ctx_name)

            g.add((win_uri, RDF.type,       EEGO["TimeWindow"]))
            g.add((win_uri, RDF.type,       TIME.Interval))
            g.add((win_uri, RDFS.label,     Literal(f"Window {window_index}")))
            start_sec = window_index * self.window_size
            g.add((win_uri, TIME.hasBeginning, Literal(start_sec, datatype=XSD.float)))
            g.add((win_uri, TIME.hasDuration,  Literal(self.window_size, datatype=XSD.float)))
            if band_name:
                band_uri = EEGO[_BAND_IRI[band_name]] if band_name in _BAND_IRI else None
                g.add((win_uri, EEGO["frequencyBand"],
                       band_uri if band_uri else Literal(band_name)))


            if self._is_single_channel:
                self._add_node_measurements(g, G, win_uri, win_idx, band_name, now)
            else:
                self._add_connectivity_edges(g, G, win_uri, win_idx, band_name, now)

    def _add_node_measurements(self, g, G, win_uri, win_idx, band_name, now):
        """Emit eego:NodeMeasurement triples for single-channel measures (e.g. PSD)."""
        for node, data in G.nodes(data=True):
            # EEGraph single-channel graphs store the value as a self-loop weight
            weight = None
            if G.has_edge(node, node):
                weight = float(G[node][node].get("weight", 0.0))
            if weight is None:
                continue
            nm_id = self._slug(self.eeg_file, win_idx, node)
            nm_uri = EEGR[f"node_measurement/{nm_id}"]
            rec_slug = self._slug(self.eeg_file)
            if self._is_bipolar_montage:
                sensor_uri = EEGR[f"channel/{rec_slug}/{node}"]
            else:
                sensor_uri = EEGR[f"electrode/{rec_slug}/{node}"]

            g.add((nm_uri, RDF.type,                EEGO["NodeMeasurement"]))
            g.add((nm_uri, RDF.type,                PROV.Entity))
            g.add((nm_uri, EEGO["ofSensor"],        sensor_uri))
            g.add((nm_uri, EEGO["value"],           Literal(str(weight), datatype=XSD.decimal)))
            g.add((nm_uri, EEGO["inWindow"],        win_uri))
            g.add((nm_uri, PROV.wasGeneratedBy,     self._activity_uri))
            g.add((nm_uri, PROV.generatedAtTime,    Literal(now, datatype=XSD.dateTime)))
            if self._measure_uri is not None:
                g.add((nm_uri, EEGO["connectivityMeasure"], self._measure_uri))
            if self._threshold_rule_uri is not None:
                g.add((nm_uri, EEGO["thresholdRule"], self._threshold_rule_uri))
            if band_name:
                band_uri = EEGO[_BAND_IRI[band_name]] if band_name in _BAND_IRI else None
                g.add((nm_uri, EEGO["frequencyBand"],
                       band_uri if band_uri else Literal(band_name)))

    def _add_connectivity_edges(self, g, G, win_uri, win_idx, band_name, now):
        """Emit eego:ConnectivityEdge triples for pairwise measures (directed or not)."""
        now_lit = Literal(now, datatype=XSD.dateTime)
        for u, v, data in G.edges(data=True):
            # Skip self-loops (can appear in some EEGraph outputs)
            if u == v:
                continue
            weight = float(data.get("weight", 0.0))
            edge_id = self._slug(self.eeg_file, win_idx, u, v)
            edge_uri = EEGR[f"edge/{edge_id}"]

            rec_slug = self._slug(self.eeg_file)
            if self._is_bipolar_montage:
                u_uri = EEGR[f"channel/{rec_slug}/{u}"]
                v_uri = EEGR[f"channel/{rec_slug}/{v}"]
            else:
                u_uri = EEGR[f"electrode/{rec_slug}/{u}"]
                v_uri = EEGR[f"electrode/{rec_slug}/{v}"]

            g.add((edge_uri, RDF.type,                EEGO["ConnectivityEdge"]))
            g.add((edge_uri, RDF.type,                PROV.Entity))
            g.add((edge_uri, EEGO["sourceElectrode"], u_uri))
            g.add((edge_uri, EEGO["targetElectrode"], v_uri))
            g.add((edge_uri, EEGO["weight"],          Literal(str(weight), datatype=XSD.decimal)))
            if self._measure_uri is not None:
                g.add((edge_uri, EEGO["connectivityMeasure"], self._measure_uri))
            else:
                g.add((edge_uri, EEGO["connectivityMeasure"], Literal(self.connectivity)))
            g.add((edge_uri, EEGO["inWindow"],        win_uri))
            g.add((edge_uri, PROV.wasGeneratedBy,     self._activity_uri))
            g.add((edge_uri, PROV.generatedAtTime,    now_lit))
            if self._threshold_rule_uri is not None:
                g.add((edge_uri, EEGO["thresholdRule"], self._threshold_rule_uri))
            if band_name:
                band_uri = EEGO[_BAND_IRI[band_name]] if band_name in _BAND_IRI else None
                g.add((edge_uri, EEGO["frequencyBand"],
                       band_uri if band_uri else Literal(band_name)))


class EEGraphKGExporter(EEGKGExporter):
    """EEGKGExporter pre-configured with EEGraph provenance metadata."""

    def __init__(
        self,
        graphs_dict: Dict[int, nx.Graph],
        ch_names: List[str],
        connectivity: str = "unknown",
        threshold: Optional[float] = None,
        bands: Optional[List] = None,
        eeg_file: str = "unknown.edf",
        window_size: int = 5,
        montage: str = "auto",
        recording_condition: Optional[str] = None,
        annotations: Optional[List[Dict]] = None,
        sample_rate: Optional[float] = None,
        segment_start: Optional[float] = None,
        segment_end: Optional[float] = None,
        dataset_url: Optional[str] = None,
    ):
        super().__init__(
            graphs_dict=graphs_dict,
            ch_names=ch_names,
            connectivity=connectivity,
            threshold=threshold,
            bands=bands,
            eeg_file=eeg_file,
            window_size=window_size,
            montage=montage,
            tool_name="EEGraph",
            tool_version="0.1.17",
            tool_url="https://github.com/ufvceiec/EEGRAPH",
            recording_condition=recording_condition,
            annotations=annotations,
            sample_rate=sample_rate,
            segment_start=segment_start,
            segment_end=segment_end,
            dataset_url=dataset_url,
        )


def export_kg(
    graphs_dict: Dict[int, nx.Graph],
    ch_names: List[str],
    connectivity: str = "unknown",
    threshold: Optional[float] = None,
    bands: Optional[List] = None,
    eeg_file: str = "unknown.edf",
    window_size: int = 5,
    output_path: Optional[str] = None,
    montage: str = "auto",
    tool_name: str = "EEGraph",
    tool_version: str = "0.1.17",
    tool_url: str = "https://github.com/ufvceiec/EEGRAPH",
) -> EEGKGExporter:
    """Build and optionally save a KG from windowed EEG connectivity data.

    Returns the populated exporter so callers can access the ConjunctiveGraph
    or serialise to additional formats.
    """
    exp = EEGKGExporter(
        graphs_dict=graphs_dict,
        ch_names=ch_names,
        connectivity=connectivity,
        threshold=threshold,
        bands=bands,
        eeg_file=eeg_file,
        window_size=window_size,
        montage=montage,
        tool_name=tool_name,
        tool_version=tool_version,
        tool_url=tool_url,
    )
    exp.build()
    if output_path:
        exp.to_turtle(output_path)
    return exp
