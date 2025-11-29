from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_STRUCTURE = {
    "meta": {"version": 1},
    # cada modelo terá uma chave (ex.: "skipgcn", "rf", "logreg", "mlp"), com {"runs": [...]}
}

class ResultsStore:
    """
    Minimal, robust results store:
    - Arquivo JSON único (ex.: data/artifacts/skipgcn_results.json)
    - Acumula execuções em 'runs' por modelo (append)
    - Cria estrutura se não existir
    - Salva de forma atômica (tmp -> rename)
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = {}
        self._load_or_init()

    # --------------- IO ---------------
    def _load_or_init(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                # fallback seguro em caso de arquivo corrompido
                self.data = DEFAULT_STRUCTURE.copy()
        else:
            self.data = DEFAULT_STRUCTURE.copy()

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # atomic on POSIX

    # --------------- API pública ---------------
    def ensure_model(self, model_key: str) -> None:
        if model_key not in self.data:
            self.data[model_key] = {"runs": []}

    def append_run(self, model_key: str, run: Dict[str, Any]) -> None:
        """
        Acrescenta uma execução (não faz deduplicação). Você pode colocar um run_id dentro de `run`
        se quiser identificar depois. Adiciona automaticamente timestamp se não existir.
        """
        self.ensure_model(model_key)
        if "timestamp" not in run:
            run["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.data[model_key]["runs"].append(run)
        self.save()

    def get_runs(self, model_key: str, feature_set: Optional[str] = None) -> List[Dict[str, Any]]:
        if model_key not in self.data:
            return []
        runs = self.data[model_key].get("runs", [])
        if feature_set is not None:
            runs = [r for r in runs if r.get("feature_set") == feature_set]
        return runs

    def last_run(self, model_key: str) -> Optional[Dict[str, Any]]:
        runs = self.get_runs(model_key)
        return runs[-1] if runs else None
