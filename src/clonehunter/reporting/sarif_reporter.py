from __future__ import annotations

import json

from clonehunter.core.types import Finding, FunctionRef, ScanResult
from clonehunter.reporting.schema import SCHEMA_VERSION


class SarifReporter:
    def write(self, result: ScanResult, out_path: str) -> None:
        runs = [
            {
                "tool": {
                    "driver": {
                        "name": "CloneHunter",
                        "informationUri": "https://example.com/clonehunter",
                        "rules": [
                            {
                                "id": "clonehunter",
                                "name": "SemanticClone",
                                "shortDescription": {"text": "Semantic code clone"},
                            }
                        ],
                    }
                },
                "results": [_to_result(f) for f in result.findings],
            }
        ]
        payload = {
            "$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json",
            "version": "2.1.0",
            "properties": {"schema_version": SCHEMA_VERSION},
            "runs": runs,
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def _to_result(finding: Finding) -> dict[str, object]:
    def loc(func: FunctionRef) -> dict[str, object]:
        return {
            "physicalLocation": {
                "artifactLocation": {"uri": func.file.path},
                "region": {"startLine": func.start_line, "endLine": func.end_line},
            }
        }

    return {
        "ruleId": "clonehunter",
        "level": "note",
        "message": {"text": "Potential semantic clone"},
        "locations": [loc(finding.function_a), loc(finding.function_b)],
        "properties": {
            "score": finding.score,
            "duplicated_lines": finding.duplicated_lines,
            "reasons": finding.reasons,
            "metadata": finding.metadata,
        },
    }
