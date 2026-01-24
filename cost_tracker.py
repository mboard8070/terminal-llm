"""
API Cost Tracking for MAUDE.

Logs all API calls and tracks daily spending with configurable limits.
"""

import os
import json
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass, asdict
from typing import Optional

from providers import PROVIDERS


@dataclass
class APICall:
    """Record of a single API call."""
    timestamp: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    agent_type: str
    task_preview: str  # First 50 chars of task


class CostTracker:
    """Track API costs with daily limits."""

    LOG_DIR = Path.home() / ".config" / "maude" / "logs"

    def __init__(self, daily_limit: float = None):
        self.daily_limit = daily_limit or float(os.environ.get("MAUDE_COST_LIMIT", "5.00"))
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._today_cache: Optional[float] = None
        self._cache_date: Optional[str] = None

    def _get_log_file(self, for_date: date = None) -> Path:
        """Get log file path for a specific date."""
        d = for_date or date.today()
        return self.LOG_DIR / f"costs_{d.isoformat()}.jsonl"

    def log_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent_type: str,
        task: str = ""
    ) -> float:
        """Log an API call and return its cost."""
        config = PROVIDERS.get(provider)
        if not config:
            return 0.0

        cost = (
            input_tokens / 1000 * config.cost_per_1k_input +
            output_tokens / 1000 * config.cost_per_1k_output
        )

        call = APICall(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            agent_type=agent_type,
            task_preview=task[:50] if task else ""
        )

        log_file = self._get_log_file()
        with open(log_file, "a") as f:
            f.write(json.dumps(asdict(call)) + "\n")

        # Invalidate cache
        self._today_cache = None

        return cost

    def get_today_spend(self) -> float:
        """Get total spend for today."""
        today = date.today().isoformat()

        # Use cache if valid
        if self._cache_date == today and self._today_cache is not None:
            return self._today_cache

        total = 0.0
        log_file = self._get_log_file()

        if log_file.exists():
            with open(log_file) as f:
                for line in f:
                    try:
                        call = json.loads(line)
                        total += call.get("cost_usd", 0.0)
                    except json.JSONDecodeError:
                        pass

        self._today_cache = total
        self._cache_date = today
        return total

    def check_limit(self) -> bool:
        """Return True if under daily limit."""
        return self.get_today_spend() < self.daily_limit

    def get_remaining(self) -> float:
        """Get remaining budget for today."""
        return max(0.0, self.daily_limit - self.get_today_spend())

    def get_summary(self) -> str:
        """Get cost summary for display."""
        today_spend = self.get_today_spend()
        remaining = self.get_remaining()

        lines = [
            f"Today's API spending: ${today_spend:.4f}",
            f"Daily limit: ${self.daily_limit:.2f}",
            f"Remaining: ${remaining:.2f}",
        ]

        if today_spend > 0:
            lines.append("")
            lines.append("Breakdown by provider:")
            breakdown = self._get_breakdown()
            for provider, data in sorted(breakdown.items(), key=lambda x: -x[1]["cost"]):
                lines.append(f"  {provider}: ${data['cost']:.4f} ({data['calls']} calls)")

        return "\n".join(lines)

    def _get_breakdown(self) -> dict:
        """Get spending breakdown by provider for today."""
        breakdown = {}
        log_file = self._get_log_file()

        if not log_file.exists():
            return breakdown

        with open(log_file) as f:
            for line in f:
                try:
                    call = json.loads(line)
                    provider = call.get("provider", "unknown")
                    if provider not in breakdown:
                        breakdown[provider] = {"cost": 0.0, "calls": 0, "tokens": 0}
                    breakdown[provider]["cost"] += call.get("cost_usd", 0.0)
                    breakdown[provider]["calls"] += 1
                    breakdown[provider]["tokens"] += call.get("input_tokens", 0) + call.get("output_tokens", 0)
                except json.JSONDecodeError:
                    pass

        return breakdown

    def warn_if_over_limit(self) -> Optional[str]:
        """Return warning message if over limit, None otherwise."""
        if not self.check_limit():
            return f"⚠ Daily cost limit (${self.daily_limit:.2f}) reached. Cloud agents disabled until tomorrow."

        remaining = self.get_remaining()
        if remaining < 1.0:
            return f"⚠ Low budget: ${remaining:.2f} remaining today"

        return None


# Global tracker instance
_tracker: Optional[CostTracker] = None


def get_tracker() -> CostTracker:
    """Get or create the global cost tracker."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


def handle_cost_command() -> str:
    """Handle /cost command."""
    return get_tracker().get_summary()
