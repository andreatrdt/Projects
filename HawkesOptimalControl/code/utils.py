from matplotlib.ticker import FuncFormatter, MultipleLocator
from config import MARKET_OPEN_SECONDS

def _sec_to_clock_str(sec: float, show_seconds: bool = False) -> str:
    """Format seconds from midnight into HH:MM (or HH:MM:SS)."""
    sec = float(sec) % 86400.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if show_seconds else f"{h:02d}:{m:02d}"

def _choose_tick_step(span_seconds: float) -> int:
    """Pick a human-friendly major tick step (in seconds) based on axis span."""
    span_seconds = abs(float(span_seconds))
    if span_seconds <= 15 * 60:
        return 60          # 1 min
    if span_seconds <= 60 * 60:
        return 5 * 60      # 5 min
    if span_seconds <= 3 * 60 * 60:
        return 15 * 60     # 15 min
    if span_seconds <= 8 * 60 * 60:
        return 30 * 60     # 30 min
    return 60 * 60         # 1 hour

def apply_time_of_day_axis(
    ax,
    *,
    base_seconds: float = 0.0,
    origin_seconds: float = MARKET_OPEN_SECONDS,
    show_seconds: bool = False,
    rotate: int = 0,
):
    """Format an x-axis that is in *seconds since market open* into wall-clock time.

    If your plotted x-values are seconds since the *episode* start, pass base_seconds=t_start.
    If your plotted x-values are already seconds since open, leave base_seconds=0.

    The displayed label is computed as origin_seconds + base_seconds + x.
    """
    try:
        xmin, xmax = ax.get_xlim()
        step = _choose_tick_step(xmax - xmin)
        ax.xaxis.set_major_locator(MultipleLocator(step))
    except Exception:
        # If locator fails for any reason, at least keep the formatter.
        pass

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: _sec_to_clock_str(origin_seconds + base_seconds + x, show_seconds=show_seconds))
    )
    if rotate:
        for lab in ax.get_xticklabels():
            lab.set_rotation(rotate)
            lab.set_horizontalalignment('right')

def t_rel(h: int, m: int, s: int = 0) -> float:
    """Seconds since market open, given a wall-clock time (HH:MM:SS)."""
    return (h * 3600 + m * 60 + s) - MARKET_OPEN_SECONDS