"""Cost dashboard.

Aggregate a usage :class:`~ai_cost_cutter.ledger.Ledger` (or a list of
:class:`~ai_cost_cutter.ledger.CallRecord`) into a spend-and-savings report you
can print, export as JSON, or render to a standalone HTML page.

Example::

    from ai_cost_cutter.dashboard import build_report
    report = build_report(ledger.records())
    print(report.render_text())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Union

from .ledger import CallRecord, Ledger


@dataclass
class ModelUsage:
    model: str
    calls: int = 0
    cost: float = 0.0
    baseline_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def savings(self) -> float:
        return self.baseline_cost - self.cost


@dataclass
class DashboardReport:
    total_calls: int = 0
    cached_calls: int = 0
    total_cost: float = 0.0
    total_baseline_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    by_model: Dict[str, ModelUsage] = field(default_factory=dict)

    @property
    def savings(self) -> float:
        return self.total_baseline_cost - self.total_cost

    @property
    def savings_pct(self) -> float:
        if self.total_baseline_cost <= 0:
            return 0.0
        return self.savings / self.total_baseline_cost

    @property
    def cache_hit_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.cached_calls / self.total_calls

    def as_dict(self) -> Dict[str, object]:
        return {
            "total_calls": self.total_calls,
            "cached_calls": self.cached_calls,
            "cache_hit_rate": self.cache_hit_rate,
            "total_cost": self.total_cost,
            "total_baseline_cost": self.total_baseline_cost,
            "savings": self.savings,
            "savings_pct": self.savings_pct,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "by_model": {
                name: {
                    "calls": u.calls,
                    "cost": u.cost,
                    "baseline_cost": u.baseline_cost,
                    "savings": u.savings,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                }
                for name, u in self.by_model.items()
            },
        }

    def render_text(self) -> str:
        lines: List[str] = []
        lines.append("ai-cost-cutter — cost dashboard")
        lines.append("=" * 38)
        lines.append(f"calls:           {self.total_calls}")
        lines.append(
            f"cache hits:      {self.cached_calls} "
            f"({self.cache_hit_rate:.0%} hit rate)"
        )
        lines.append(f"tokens in/out:   {self.total_input_tokens} / {self.total_output_tokens}")
        lines.append("")
        lines.append(f"actual spend:    ${self.total_cost:.4f}")
        lines.append(f"baseline spend:  ${self.total_baseline_cost:.4f}")
        lines.append(
            f"saved:           ${self.savings:.4f} ({self.savings_pct:.0%})"
        )
        if self.by_model:
            lines.append("")
            name_w = max(len(n) for n in self.by_model)
            name_w = max(name_w, len("model"))
            lines.append(
                f"{'model'.ljust(name_w)}  {'calls':>6}  {'cost':>10}  {'saved':>10}"
            )
            lines.append("-" * (name_w + 32))
            for name in sorted(self.by_model):
                u = self.by_model[name]
                lines.append(
                    f"{name.ljust(name_w)}  {u.calls:>6}  "
                    f"${u.cost:>9.4f}  ${u.savings:>9.4f}"
                )
        return "\n".join(lines)

    def render_html(self) -> str:
        rows = "".join(
            "<tr><td>{m}</td><td>{c}</td><td>${cost:.4f}</td>"
            "<td>${sv:.4f}</td></tr>".format(
                m=name, c=u.calls, cost=u.cost, sv=u.savings
            )
            for name, u in sorted(self.by_model.items())
        )
        css = (
            "body{font-family:system-ui,sans-serif;margin:2rem;color:#1a1a1a}"
            "table{border-collapse:collapse;margin-top:1rem}"
            "td,th{border:1px solid #ddd;padding:.4rem .8rem;text-align:right}"
            "td:first-child,th:first-child{text-align:left}"
            ".big{font-size:1.6rem;font-weight:600}"
            ".save{color:#0a7d28}"
        )
        summary = (
            "<p class='big'>Saved <span class='save'>${sv:.4f}</span> "
            "({pct:.0%}) &mdash; ${cost:.4f} spent vs ${base:.4f} baseline</p>"
            "<p>{calls} calls, {hits} cache hits ({hr:.0%} hit rate)</p>"
        ).format(
            sv=self.savings,
            pct=self.savings_pct,
            cost=self.total_cost,
            base=self.total_baseline_cost,
            calls=self.total_calls,
            hits=self.cached_calls,
            hr=self.cache_hit_rate,
        )
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>ai-cost-cutter dashboard</title>"
            "<style>" + css + "</style></head><body>"
            "<h1>Cost dashboard</h1>" + summary
            + "<table><thead><tr><th>model</th><th>calls</th><th>cost</th>"
            "<th>saved</th></tr></thead><tbody>" + rows + "</tbody></table>"
            "</body></html>"
        )


def build_report(
    source: Union[Ledger, Iterable[CallRecord]]
) -> DashboardReport:
    """Build a :class:`DashboardReport` from a ledger or records."""
    records = source.records() if isinstance(source, Ledger) else list(source)
    report = DashboardReport()
    for rec in records:
        report.total_calls += 1
        if rec.cached:
            report.cached_calls += 1
        report.total_cost += rec.cost
        report.total_baseline_cost += rec.effective_baseline
        report.total_input_tokens += rec.input_tokens
        report.total_output_tokens += rec.output_tokens

        usage = report.by_model.get(rec.model)
        if usage is None:
            usage = ModelUsage(model=rec.model)
            report.by_model[rec.model] = usage
        usage.calls += 1
        usage.cost += rec.cost
        usage.baseline_cost += rec.effective_baseline
        usage.input_tokens += rec.input_tokens
        usage.output_tokens += rec.output_tokens
    return report
