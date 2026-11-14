"""The workflows dependency rule: shared modules, campaigns, no cross-campaign imports.

Layout (D2): modules directly under `drmdp.workflows` are shared; each
folder under `drmdp.workflows.control` and
`drmdp.workflows.reward_estimation.<campaign>` is a campaign. A campaign
module may import the `drmdp` library, shared workflow modules, and its
own campaign. Forbidden: an import of one campaign package from another
(including across the `reward_estimation/` domain) and an import of a
campaign package from a shared module.
"""

import ast
import pathlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

WORKFLOWS_ROOT = pathlib.Path("src/drmdp/workflows")
WORKFLOWS_PACKAGE = "drmdp.workflows"
CAMPAIGN_PACKAGES: Mapping[str, str | None] = {
    # workflows subpackage -> owning campaign; None marks shared scope
    "": None,
    "control": "control",
    "reward_estimation": None,
    "reward_estimation.encodings": "reward_estimation.encodings",
    "reward_estimation.methods": "reward_estimation.methods",
    "reward_estimation.bias_variance": "reward_estimation.bias_variance",
}


@dataclass(frozen=True)
class WorkflowModule:
    name: str
    path: pathlib.Path
    is_package: bool
    campaign: str | None


def _iter_modules(root: pathlib.Path) -> Iterator[WorkflowModule]:
    for path in sorted(root.rglob("*.py")):
        parts = list(path.with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts = parts[:-1]
        name = ".".join([WORKFLOWS_PACKAGE, *parts])
        rest = name[len(WORKFLOWS_PACKAGE) + 1 :]
        matches = [
            (len(package), campaign)
            for package, campaign in CAMPAIGN_PACKAGES.items()
            if rest == package or rest.startswith(package + ".")
        ]
        campaign = max(matches)[1] if matches else None
        yield WorkflowModule(name, path, is_package, campaign)


def _import_targets(node: ast.AST, source: WorkflowModule) -> Iterator[str]:
    """Fully-qualified module names referenced by one import statement."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            parts = source.name.split(".")
            if not source.is_package:
                parts = parts[:-1]
            parts = parts[: len(parts) - (node.level - 1)] or ["drmdp"]
            base = ".".join(parts)
            if node.module:
                base = f"{base}.{node.module}"
        else:
            base = node.module or ""
        for alias in node.names:
            yield f"{base}.{alias.name}"


class TestDependencyRule:
    def test_no_campaign_imports_from_another_campaign_or_from_shared(self):
        modules = list(_iter_modules(WORKFLOWS_ROOT))
        known = {module.name for module in modules}
        violations: list[str] = []
        for source in modules:
            tree = ast.parse(source.path.read_text(), str(source.path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                for target in _import_targets(node, source):
                    # climb to the longest prefix that is a workflows module
                    while target and target not in known:
                        target = target.rpartition(".")[0]
                    if not target.startswith(WORKFLOWS_PACKAGE):
                        continue
                    target_campaign = next(
                        module.campaign for module in modules if module.name == target
                    )
                    if target_campaign is None:
                        # shared modules may import each other
                        continue
                    if source.campaign is None or target_campaign != source.campaign:
                        violations.append(f"{source.name} -> {target}")
        assert violations == []
