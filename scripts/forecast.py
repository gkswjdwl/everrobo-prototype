"""이상치 표시 + 3개월 ARIMA 예측 (collect.py 가 호출합니다)

- flag_outliers : 같은 단지·같은 크기의 ±120일 이웃 거래 중앙값과 25% 넘게 벗어난 거래에 out=True
- build_forecast: 월평균가(이상치 제외, 진행 중인 이번 달 제외)에 ARIMA 를 맞춰 3개월 뒤까지 예측 + 80% 범위 + 과거 검증 오차

예측은 통계 모델의 단순 추정치이며 참고용입니다.
"""
import warnings
from bisect import bisect_left, bisect_right
from datetime import date
from statistics import median

import numpy as np

OUT_WINDOW_DAYS = 120   # 이웃으로 보는 기간(±일)
OUT_MIN_NEIGHBORS = 5   # 이웃이 이보다 적으면 판단 보류(표시 안 함)
OUT_THRESHOLD = 0.25    # 중앙값에서 25% 이상 벗어나면 이상치 후보

HORIZON = 3             # 예측 개월 수
WINDOW_MONTHS = 48      # 모델에 쓰는 최근 개월 수
MIN_DATA_MONTHS = 24    # 이 중 실제 거래가 있는 달이 최소 몇 개여야 하는지
MAX_GAP = 3             # 거래 없는 달이 연속 이 개월 넘으면 예측 안 함
ORDERS = [(1, 1, 0), (0, 1, 1), (1, 1, 1)]   # 추세(드리프트)는 넣지 않음: 과거 검증에서 넣은 쪽이 더 부정확했음
MAX_MOVE = 0.20         # 3개월 변화가 이보다 크면 비현실적이라 보고 예측 안 함


def _ord(d):
    y, m, dd = d.split("-")
    return date(int(y), int(m), int(dd)).toordinal()


def flag_outliers(txs):
    """각 거래에 t['out'] (bool) 을 채운다."""
    groups = {}
    for t in txs:
        t["out"] = False
        groups.setdefault((t["c"], t["size"]), []).append(t)
    for items in groups.values():
        items.sort(key=lambda t: t["date"])
        days = [_ord(t["date"]) for t in items]
        prices = [t["price"] for t in items]
        for i, t in enumerate(items):
            lo = bisect_left(days, days[i] - OUT_WINDOW_DAYS)
            hi = bisect_right(days, days[i] + OUT_WINDOW_DAYS)
            nb = prices[lo:i] + prices[i + 1:hi]
            if len(nb) < OUT_MIN_NEIGHBORS:
                continue
            med = median(nb)
            if med > 0 and abs(t["price"] / med - 1) > OUT_THRESHOLD:
                t["out"] = True
    return txs


def _ym_index(ym):
    return int(ym[:4]) * 12 + int(ym[5:7]) - 1


def _ym_str(idx):
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def _monthly(items):
    """[(ym_index, 평균가)] 오름차순. 이상치는 제외."""
    acc = {}
    for t in items:
        if t.get("out"):
            continue
        acc.setdefault(_ym_index(t["date"][:7]), []).append(t["price"])
    return [(k, sum(v) / len(v)) for k, v in sorted(acc.items())]


def _series(monthly, now_idx):
    """최근 WINDOW_MONTHS 개월을 달마다 하나씩, 빈 달은 선형보간. 조건 미달이면 None."""
    monthly = [(k, v) for k, v in monthly if k < now_idx]   # 이번 달은 아직 거래가 덜 모여 제외
    if not monthly:
        return None
    last = monthly[-1][0]
    if now_idx - last > 3:              # 최근 거래가 너무 오래됨
        return None
    first = max(monthly[0][0], last - WINDOW_MONTHS + 1)
    pts = [(k, v) for k, v in monthly if k >= first]
    if len(pts) < MIN_DATA_MONTHS:
        return None
    gap = max(b[0] - a[0] for a, b in zip(pts, pts[1:])) if len(pts) > 1 else 0
    if gap - 1 > MAX_GAP:
        return None
    xs = np.arange(first, last + 1)
    y = np.interp(xs, [k for k, _ in pts], [np.log(v) for _, v in pts])
    return first, last, y


def _fit(y, order):
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(y, order=order, trend="n").fit()


def _predict(y, steps, order=None):
    """order 가 없으면 AIC 로 고른다. (예측 평균, 하한, 상한, 사용 order) — 로그 스케일."""
    best = None
    for o in ([order] if order else ORDERS):
        try:
            r = _fit(y, o)
        except Exception:
            continue
        if not np.isfinite(r.aic):
            continue
        if best is None or r.aic < best[0].aic:
            best = (r, o)
    if best is None:
        return None
    r, o = best
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = r.get_forecast(steps)
        mean = np.asarray(fc.predicted_mean)
        ci = np.asarray(fc.conf_int(alpha=0.2))
    return mean, ci[:, 0], ci[:, 1], o


def _backtest(y, order):
    """최근 여러 시점에서 '3개월 뒤'를 맞혀 보고 평균 절대 오차율(%)을 구한다. 단순 비교용 naive(마지막 값 유지)도 함께."""
    errs, nerrs = [], []
    for off in range(HORIZON, HORIZON + 8):
        cut = len(y) - off
        if cut < 20:
            break
        res = _predict(y[:cut], HORIZON, order)
        if res is None:
            continue
        actual = np.exp(y[cut + HORIZON - 1])
        errs.append(abs(np.exp(res[0][-1]) / actual - 1))
        nerrs.append(abs(np.exp(y[cut - 1]) / actual - 1))
    if len(errs) < 3:
        return None, None
    return round(float(np.mean(errs)) * 100, 1), round(float(np.mean(nerrs)) * 100, 1)


def forecast_one(items, now_idx):
    s = _series(_monthly(items), now_idx)
    if s is None:
        return None
    first, last, y = s
    res = _predict(y, HORIZON)
    if res is None:
        return None
    mean, lo, hi, order = res
    base = float(np.exp(y[-1]))
    if abs(np.exp(mean[-1]) / base - 1) > MAX_MOVE:
        return None
    mape, naive = _backtest(y, order)
    return {
        "base": _ym_str(last),
        "base_p": int(round(base)),
        "pts": [{"m": _ym_str(last + i + 1), "p": int(round(float(np.exp(mean[i])))),
                 "lo": int(round(float(np.exp(lo[i])))), "hi": int(round(float(np.exp(hi[i]))))}
                for i in range(HORIZON)],
        "order": list(order),
        "mape": mape,
        "naive": naive,
        "n": len(y),
    }


def build_forecast(txs, now_ym):
    """{단지id: {크기: 예측 dict}} — 예측할 수 없는 조합은 빠진다."""
    now_idx = _ym_index(now_ym)
    groups = {}
    for t in txs:
        groups.setdefault((t["c"], t["size"]), []).append(t)
    out = {}
    for (c, size), items in sorted(groups.items()):
        try:
            f = forecast_one(items, now_idx)
        except Exception:      # 예측 실패가 수집 전체를 막지 않도록
            f = None
        if f:
            out.setdefault(c, {})[str(size)] = f
    return out
