from clonehunter.engines.semantic_engine import SemanticEngine
from clonehunter.engines.sonarqube_engine import SonarQubeEngine
from clonehunter.model.registry import register_engine

register_engine("semantic", lambda: SemanticEngine())
register_engine("sonarqube", lambda: SonarQubeEngine())

__all__ = ["SemanticEngine", "SonarQubeEngine"]
