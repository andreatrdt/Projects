# event_cleaning.py
import numpy as np

def _as_sorted_float(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    return np.sort(x)

def collapse_same_timestamp(t: np.ndarray, tol: float = 0.0):
    """
    Collassa eventi con timestamp uguale (o entro tol) in un solo evento.
    Ritorna:
      t_unique: array di tempi unici (ordinati)
      marks: conteggi per ogni tempo (>=1)
    """
    t = _as_sorted_float(t)
    if t.size == 0:
        return t, np.array([], dtype=int)

    if tol <= 0.0:
        tu, counts = np.unique(t, return_counts=True)
        return tu, counts.astype(int)

    # raggruppa consecutivi entro tol
    tu = [t[0]]
    counts = [1]
    for x in t[1:]:
        if abs(x - tu[-1]) <= tol:
            counts[-1] += 1
        else:
            tu.append(x)
            counts.append(1)
    return np.array(tu, float), np.array(counts, int)

def collapse_events_dict(events: dict, components: list[str], tol: float = 0.0):
    """
    Applica collapse_same_timestamp a ciascuna componente.
    """
    out = {}
    marks = {}
    for name in components:
        tu, c = collapse_same_timestamp(events.get(name, np.array([])), tol=tol)
        out[name] = tu
        marks[name] = c
    return out, marks

def _auto_eps_global(times_all: np.ndarray) -> float:
    """
    Epsilon per tie-break globale: molto più piccolo del minimo gap positivo.
    """
    t = _as_sorted_float(times_all)
    if t.size < 2:
        return 1e-9
    u = np.unique(t)
    du = np.diff(u)
    du = du[du > 0]
    if du.size == 0:
        return 1e-9
    dt_min = float(du.min())
    # prendiamo un epsilon che sia "invisibile" rispetto alla risoluzione
    return max(dt_min * 1e-3, 1e-12)  # 0.1% del minimo gap positivo

def break_global_ties(events: dict, components: list[str], eps: float | str = "auto"):
    """
    Assicura che NON esistano timestamp identici tra componenti diverse.
    Fa un tie-break deterministico: per ogni gruppo con lo stesso timestamp,
    ordina per comp index e aggiunge k*eps.
    """
    # merge
    times = []
    comps = []
    for j, name in enumerate(components):
        tj = _as_sorted_float(events.get(name, np.array([])))
        times.append(tj)
        comps.append(np.full_like(tj, j, dtype=int))
    if len(times) == 0:
        return events, 0.0

    t = np.concatenate(times) if times else np.array([], float)
    c = np.concatenate(comps) if comps else np.array([], int)
    if t.size == 0:
        return events, 0.0

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    c = c[order]

    eps_val = _auto_eps_global(t) if eps == "auto" else float(eps)

    # tie-break
    t2 = t.copy()
    i = 0
    while i < t2.size:
        j = i + 1
        while j < t2.size and t2[j] == t2[i]:
            j += 1
        if j - i > 1:
            # group [i, j)
            # ordina deterministicamente per componente
            idx = np.arange(i, j)
            idx_sorted = idx[np.argsort(c[i:j], kind="mergesort")]
            for k, idxk in enumerate(idx_sorted):
                t2[idxk] += k * eps_val
        i = j

    # split back
    out = {name: [] for name in components}
    for tk, jk in zip(t2, c):
        out[components[jk]].append(tk)
    for name in components:
        out[name] = np.array(out[name], dtype=float)
        out[name].sort()

    return out, eps_val

def dt0_stats(t: np.ndarray):
    t = _as_sorted_float(t)
    if t.size < 2:
        return {"n": int(t.size), "share_dt0": np.nan}
    dt = np.diff(t)
    return {"n": int(t.size), "share_dt0": float(np.mean(dt == 0.0))}

def prepare_hawkes_events(events_raw: dict, components: list[str],
                         collapse_tol: float = 0.0,
                         global_tie_break: bool = True,
                         eps: float | str = "auto"):
    """
    Pipeline completa:
      1) collassa duplicati intra-componente (t uguali -> 1 evento) + marks
      2) tie-break globale tra componenti (se richiesto)
    """
    events1, marks = collapse_events_dict(events_raw, components, tol=collapse_tol)

    eps_used = 0.0
    events2 = events1
    if global_tie_break:
        events2, eps_used = break_global_ties(events1, components, eps=eps)

    meta = {
        "eps_used": eps_used,
        "dt0_after": {name: dt0_stats(events2[name]) for name in components},
        "n_raw": {name: int(len(events_raw.get(name, []))) for name in components},
        "n_collapsed": {name: int(len(events1[name])) for name in components},
    }
    return events2, marks, meta
