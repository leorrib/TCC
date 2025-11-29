# src/models/oc/factory.py
from __future__ import annotations
from typing import Dict, Any
from .architectures import OCGAT, OCGCN, OCGraphSAGE

class OCFactory:
    @staticmethod
    def build(kind: str, in_channels: int, cfg: Dict[str, Any]):
        kind = kind.lower()
        common = cfg["models"]["ocgnn"]["common"]
        if kind == "ocgat":
            p = cfg["models"]["ocgnn"]["ocgat"]
            return OCGAT(in_channels, hidden=p["hidden"], layers=p["layers"], heads=p["heads"], dropout=common["dropout"])
        elif kind == "ocgcn":
            p = cfg["models"]["ocgnn"]["ocgcn"]
            return OCGCN(in_channels, hidden=p["hidden"], layers=p["layers"], dropout=common["dropout"])
        elif kind == "ocgraphsage":
            p = cfg["models"]["ocgnn"]["ocgraphsage"]
            return OCGraphSAGE(in_channels, hidden=p["hidden"], layers=p["layers"], agg=p["agg"], dropout=common["dropout"])
        else:
            raise ValueError(f"OCGNN desconhecido: {kind}")
