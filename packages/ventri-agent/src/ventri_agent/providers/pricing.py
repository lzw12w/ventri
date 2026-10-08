"""Price tables and DeepSeek's peak / off-peak schedule (DESIGN.md 5.2).

Prices are data (configurable), never logic. Verified 2026-10-08 against
https://api-docs.deepseek.com/quick_start/pricing: USD per 1M tokens at *peak*;
off-peak is half. Peak = 01:00-04:00 and 06:00-10:00 UTC, Monday-Friday,
excluding Chinese public holidays (Beijing 09:00-12:00 and 14:00-18:00).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone

from ..messages import Money, Usage

BEIJING = timezone(timedelta(hours=8), "CST")


@dataclass(frozen=True)
class ModelPrice:
    """Peak prices, USD per 1M tokens."""

    cache_hit: float
    cache_miss: float
    output: float


DEEPSEEK_PRICES: dict[str, ModelPrice] = {
    "deepseek-flash": ModelPrice(0.006, 0.30, 1.20),
    "deepseek-v4-pro": ModelPrice(0.044, 1.32, 3.96),
}
# Legacy names still accepted by the API and billed at the Flash price.
DEEPSEEK_ALIASES = {"deepseek-v4-flash": "deepseek-flash", "deepseek-v4-flash-vision-exp": "deepseek-flash"}


@dataclass
class PeakSchedule:
    windows_utc: tuple[tuple[int, int], ...] = ((1, 4), (6, 10))
    weekdays: frozenset[int] = frozenset(range(5))  # Monday=0 .. Friday=4
    holidays: frozenset[date] = frozenset()          # Chinese public holidays (Beijing dates)
    off_peak_factor: float = 0.5

    def is_peak(self, at: datetime) -> bool:
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        u = at.astimezone(UTC)
        if at.astimezone(BEIJING).date() in self.holidays or u.weekday() not in self.weekdays:
            return False
        return any(a <= u.hour < b for a, b in self.windows_utc)

    def next_off_peak(self, at: datetime) -> datetime:
        """The first off-peak instant at or after ``at`` (for deferrable work)."""
        t = at if at.tzinfo else at.replace(tzinfo=UTC)
        for _ in range(24 * 8):
            if not self.is_peak(t):
                return t
            t = (t.astimezone(UTC) + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        return t


@dataclass
class PriceTable:
    prices: dict[str, ModelPrice] = field(default_factory=lambda: dict(DEEPSEEK_PRICES))
    aliases: dict[str, str] = field(default_factory=lambda: dict(DEEPSEEK_ALIASES))
    schedule: PeakSchedule = field(default_factory=PeakSchedule)

    @classmethod
    def from_config(cls, prices: dict | None = None, holidays: Iterable[date | str] = ()) -> PriceTable:
        t = cls()
        for model, p in (prices or {}).items():
            t.prices[model] = p if isinstance(p, ModelPrice) else ModelPrice(**p)
        hs = frozenset(h if isinstance(h, date) else date.fromisoformat(h) for h in holidays)
        t.schedule = PeakSchedule(holidays=hs)
        return t

    def lookup(self, model: str) -> ModelPrice | None:
        return self.prices.get(self.aliases.get(model, model))

    def price(self, usage: Usage, at: datetime, model: str) -> Money:
        p = self.lookup(model)
        if p is None:
            return Money(0.0)
        f = 1.0 if self.schedule.is_peak(at) else self.schedule.off_peak_factor
        usd = (usage.cache_hit * p.cache_hit + usage.cache_miss * p.cache_miss
               + usage.completion_tokens * p.output) / 1_000_000 * f
        return Money(usd)
