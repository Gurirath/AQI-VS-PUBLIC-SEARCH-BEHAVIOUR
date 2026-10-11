import sys

import pandas as pd

from . import awareness as A
from . import causal as C
from . import features as F
from . import stats as S
from .config import CITIES, PROCESSED

ROLES = ("attention", "health")
SAVED = ("results_summary", "awareness_scorecard", "verdicts")


def _stack(frames):
    out = pd.concat(frames, names=["role"]).reset_index()
    out.insert(2, "series", [F.role_label(c, r) for c, r in zip(out["city"], out["role"])])
    out = out[["city", *out.columns.drop("city")]]
    return out.sort_values("city", key=lambda s: s.map(list(CITIES).index), kind="stable").reset_index(drop=True)


def build(long):
    corr = S.correlation_table(long)
    cards, grangers, skills, verdicts, predictions = {}, {}, {}, {}, {}
    for role in ROLES:
        cards[role] = A.scorecard(long, role)
        grangers[role] = C.granger_table(long, role)
        skills[role], predictions[role] = C.forecast_skill(long, role=role)
        verdicts[role] = C.verdicts(corr, grangers[role], skills[role], cards[role], role)
    return {"results_summary": corr, "awareness_scorecard": _stack(cards), "verdicts": _stack(verdicts),
            "granger": pd.concat(grangers, names=["role"]).reset_index(level=0).reset_index(drop=True),
            "forecast_skill": _stack(skills), "predictions": predictions}


def write(tables):
    for name in SAVED:
        tables[name].to_csv(PROCESSED / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(tables[name])} rows")
    return tables


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return write(build(F.load_long()))


if __name__ == "__main__":
    main()
