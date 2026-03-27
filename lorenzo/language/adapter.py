from __future__ import annotations

from lorenzo.language.backends import EchoBackend, LanguageBackend, RuleBasedBackend


class LanguageAdapter:
    def __init__(self, backend: LanguageBackend) -> None:
        self.backend = backend

    @classmethod
    def from_name(cls, backend_name: str) -> "LanguageAdapter":
        mapping: dict[str, LanguageBackend] = {
            "rule_based": RuleBasedBackend(),
            "echo": EchoBackend(),
        }
        selected = mapping.get(backend_name)
        if selected is None:
            raise ValueError(f"Unknown backend '{backend_name}'. Available: {', '.join(mapping)}")
        return cls(selected)

    def generate(self, *args, **kwargs) -> str:
        return self.backend.generate(*args, **kwargs)
