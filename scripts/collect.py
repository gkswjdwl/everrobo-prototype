"""잠실 대단지(잠실엘스·리센츠·트리지움) 매매 실거래가 수집기
(국토교통부 '아파트 매매 실거래가 상세 자료' Open API)

세 단지 모두 송파구 잠실동이라 한 번의 호출(송파구 + 계약년월)로 함께 받아 단지별로 나눕니다.
수집한 뒤 이상치 표시(out)와 3개월 ARIMA 예측(forecast)을 같은 JSON 파일에 함께 저장합니다. (forecast.py)

사용법
  python scripts/collect.py --probe          # 인증키·응답 구조·단지명 매칭 점검 (저장 안 함)
  python scripts/collect.py --since 200801   # 최초 1회: 2008년 1월부터 전체 수집
  python scripts/collect.py                  # 정기 갱신: 최근 3개월을 다시 받아 덮어쓰기 (확인 시각은 status.json 에 기록)

인증키는 환경변수 DATA_GO_KR_KEY 또는 프로젝트 루트의 .env 파일에서 읽습니다.
"""
import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecast as fc_mod  # noqa: E402

BASE_URL = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"
LAWD_CD = "11710"  # 송파구 법정동코드 앞 5자리
DONG = "잠실동"
PAGE_SIZE = 1000

# 수집 대상 단지. match 는 단지명(공백 제거)에 포함되어야 하는 글자입니다.
COMPLEXES = [
    {"id": "els", "name": "잠실엘스", "match": ["잠실엘스"]},
    {"id": "ricenz", "name": "리센츠", "match": ["리센츠"]},
    {"id": "trizium", "name": "트리지움", "match": ["트리지움"]},
]
DEFAULT_COMPLEX = "els"  # 예전(v1) 파일은 잠실엘스만 담고 있었음

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "transactions.json"
STATUS = ROOT / "docs" / "data" / "status.json"   # 마지막으로 확인한 시각만 담는 작은 파일
KST = timezone(timedelta(hours=9))

FIELD_ORDER = ("c", "date", "dong", "floor", "area", "size", "price", "direct", "built", "out")

# 응답에 반드시 있어야 하는 태그 (없으면 API 스펙이 바뀐 것이므로 중단)
REQUIRED = ("aptNm", "umdNm", "dealAmount", "excluUseAr", "floor",
            "dealYear", "dealMonth", "dealDay")


# ---------------------------------------------------------------- 인증키
def load_key():
    key = os.environ.get("DATA_GO_KR_KEY", "").strip()
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DATA_GO_KR_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("DATA_GO_KR_KEY 가 없습니다. 프로젝트 루트의 .env 파일에 "
                 "DATA_GO_KR_KEY=발급받은키 를 적거나 GitHub Secrets 에 등록하세요.")
    # 포털의 'Encoding 키'를 넣어도 동작하도록 한 번 풀어 둔다 (requests 가 다시 인코딩함)
    return unquote(key)


# ---------------------------------------------------------------- API 호출
def request_xml(params, tries=3):
    last = "알 수 없는 오류"
    for i in range(tries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            # 주의: 예외 메시지에는 URL(=인증키)이 들어 있으므로 출력하지 않는다
            status = getattr(e.response, "status_code", None)
            last = f"{type(e).__name__}" + (f" (HTTP {status})" if status else "")
            time.sleep(2 ** i)
            continue
        try:
            return ET.fromstring(r.content)
        except ET.ParseError:
            raise RuntimeError(f"XML 이 아닌 응답을 받았습니다: {r.text[:200]!r}")
    raise RuntimeError(f"API 호출 실패: {last}")


def fetch_month(key, ym):
    """한 달(계약년월)치 송파구 전체 거래를 페이지 끝까지 받아 dict 리스트로 반환."""
    items, page = [], 1
    while True:
        root = request_xml({
            "serviceKey": key, "LAWD_CD": LAWD_CD, "DEAL_YMD": ym,
            "pageNo": page, "numOfRows": PAGE_SIZE,
        })
        if root.tag == "OpenAPI_ServiceResponse":  # 인증키 오류 등 게이트웨이 에러
            reason = root.findtext(".//returnAuthMsg") or root.findtext(".//errMsg")
            raise RuntimeError(f"API 게이트웨이 오류: {reason}")
        code = (root.findtext(".//resultCode") or "").strip()
        if code not in ("00", "000"):
            raise RuntimeError(f"API 오류 {code}: {root.findtext('.//resultMsg')}")
        total = int(root.findtext(".//totalCount") or 0)
        for el in root.iterfind(".//item"):
            items.append({c.tag: (c.text or "").strip() for c in el})
        if page * PAGE_SIZE >= total:
            return items
        page += 1
        time.sleep(0.2)


def check_schema(item):
    missing = [t for t in REQUIRED if t not in item]
    if missing:
        sys.exit(f"응답에 예상한 필드가 없습니다: {missing}\n실제 필드: {sorted(item)}")


# ---------------------------------------------------------------- 전처리
def find_complex(apt_name):
    """단지명 -> 단지 id (대상이 아니면 None)."""
    name = apt_name.replace(" ", "")
    for c in COMPLEXES:
        if any(m in name for m in c["match"]):
            return c["id"]
    return None


def to_record(it):
    """원본 1건 -> 앱용 레코드. 대상 단지가 아니거나 해제된 거래면 None."""
    cid = find_complex(it["aptNm"])
    if cid is None:
        return None
    if it["umdNm"] and it["umdNm"] != DONG:
        return None
    if it.get("cdealType") == "O":  # 계약 해제된 거래는 제외
        return None
    area = float(it["excluUseAr"])
    built = it.get("buildYear", "")
    return {
        "c": cid,
        "date": f"{int(it['dealYear']):04d}-{int(it['dealMonth']):02d}-{int(it['dealDay']):02d}",
        "dong": it.get("aptDong", ""),            # 동 정보는 등기 완료 건에만 공개됨
        "floor": int(it["floor"]),
        "area": round(area, 2),                   # 전용면적(㎡)
        "size": int(area),                        # 평형 구간: 84.97 -> 84
        "price": int(it["dealAmount"].replace(",", "").replace(" ", "")),  # 만원
        "direct": it.get("dealingGbn") == "직거래",  # 이상치 후보 플래그
        "built": int(built) if built.isdigit() else None,
    }


def dedupe(records):
    seen, out = set(), []
    for r in records:
        k = (r["c"], r["date"], r["dong"], r["floor"], r["area"], r["price"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


# ---------------------------------------------------------------- 저장
def sort_key(t):
    return (t["date"], t["price"], t["floor"], t["c"])


def load_existing():
    """(기존 거래 목록, 기존 예측, 현재 스키마 여부). v1 파일의 거래는 잠실엘스로 간주."""
    if not OUT.exists():
        return [], {}, True
    try:
        data = json.loads(OUT.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [], {}, True
    txs = data.get("transactions", [])
    for t in txs:
        t.setdefault("c", DEFAULT_COMPLEX)
    return txs, data.get("forecast", {}), ("complexes" in data and "forecast" in data)


def write_status(now):
    """수집이 성공적으로 끝날 때마다 '확인한 시각'만 따로 기록 (새 거래가 없어도 갱신됨)."""
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({"checked": now.isoformat(timespec="seconds")}) + "\n", encoding="utf-8")


def write_output(txs, fc, now):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts, outs = {}, {}
    for t in txs:
        counts[t["c"]] = counts.get(t["c"], 0) + 1
        outs[t["c"]] = outs.get(t["c"], 0) + (1 if t.get("out") else 0)
    comps = [{"id": c["id"], "name": c["name"], "count": counts.get(c["id"], 0), "out": outs.get(c["id"], 0)}
             for c in COMPLEXES]
    lines = ",\n".join(
        "  " + json.dumps({k: t.get(k) for k in FIELD_ORDER}, ensure_ascii=False, separators=(",", ":"))
        for t in txs)
    text = ('{"unit":"만원","updated":%s,"count":%d,"complexes":%s,"forecast":%s,"transactions":[\n%s\n]}\n'
            % (json.dumps(now.isoformat(timespec="seconds")), len(txs),
               json.dumps(comps, ensure_ascii=False, separators=(",", ":")),
               json.dumps(fc, ensure_ascii=False, separators=(",", ":")), lines))
    OUT.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- 실행
def month_list(since, until):
    y, m = int(since[:4]), int(since[4:])
    uy, um = int(until[:4]), int(until[4:])
    out = []
    while (y, m) <= (uy, um):
        out.append(f"{y:04d}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def recent_months(until, n):
    y, m = int(until[:4]), int(until[4:])
    out = []
    for _ in range(n):
        out.append(f"{y:04d}{m:02d}")
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    return out[::-1]


def summarize(recs):
    return " / ".join(f"{c['name']} {sum(1 for r in recs if r['c'] == c['id'])}"
                      for c in COMPLEXES)


def probe(key, this_ym):
    ym = recent_months(this_ym, 2)[0]  # 지난달 (거래가 확실히 있는 달)
    items = fetch_month(key, ym)
    print(f"[{ym}] 송파구 전체 {len(items)}건 수신")
    if not items:
        print("응답은 정상이나 데이터가 없습니다. 인증키 승인 직후라면 잠시 후 다시 시도하세요.")
        return
    print("필드:", sorted(items[0]))
    check_schema(items[0])
    names = sorted({it["aptNm"] for it in items if it.get("umdNm") == DONG})
    print(f"{DONG} 단지명 {len(names)}개:", ", ".join(names))
    recs = [r for r in map(to_record, items) if r]
    for c in COMPLEXES:
        hit = [r for r in recs if r["c"] == c["id"]]
        print(f"  {c['name']}: {len(hit)}건 매칭" + ("" if hit else "  <- 0건 (그 달 거래가 없었거나 단지명이 다를 수 있음)"))
    if recs:
        print("예시:", recs[0])


def main():
    ap = argparse.ArgumentParser(description="잠실 대단지 실거래가 수집기")
    ap.add_argument("--probe", action="store_true", help="키/응답 구조/단지명 매칭만 점검")
    ap.add_argument("--since", help="이 연월(YYYYMM)부터 현재까지 전체 수집 (최초 1회용)")
    ap.add_argument("--months", type=int, default=3, help="정기 갱신 시 다시 받을 최근 개월 수")
    args = ap.parse_args()

    key = load_key()
    now = datetime.now(KST)
    this_ym = now.strftime("%Y%m")

    if args.probe:
        probe(key, this_ym)
        return

    if args.since:
        if len(args.since) != 6 or not args.since.isdigit():
            sys.exit("--since 는 YYYYMM 형식이어야 합니다. 예: --since 200801")
        months = month_list(args.since, this_ym)
    else:
        months = recent_months(this_ym, max(1, args.months))

    fresh, checked = {}, False
    for ym in months:
        items = fetch_month(key, ym)
        if items and not checked:
            check_schema(items[0])
            checked = True
        recs = dedupe([r for r in map(to_record, items) if r])
        fresh[ym] = recs
        print(f"{ym}: 송파구 {len(items)}건 -> {summarize(recs)}")
        time.sleep(0.2)

    # 다시 받은 달은 통째로 교체 (신고 지연·정정·해제를 자동 반영)
    old, old_fc, schema_ok = load_existing()
    kept = [t for t in old if t["date"][:7].replace("-", "") not in fresh]
    merged = sorted(kept + [r for recs in fresh.values() for r in recs], key=sort_key, reverse=True)

    # 이상치 표시와 예측은 전체 데이터로 매번 다시 계산 (실패해도 수집 결과는 저장)
    fc_mod.flag_outliers(merged)
    try:
        fc = fc_mod.build_forecast(merged, now.strftime("%Y-%m"))
    except Exception as e:
        print(f"예측 계산 실패({type(e).__name__}) - 예측 없이 저장합니다")
        fc = old_fc

    if schema_ok and fc == old_fc and merged == sorted(old, key=sort_key, reverse=True):
        write_status(now)
        print("변경 없음 (확인 시각만 기록)")
        return
    write_output(merged, fc, now)
    write_status(now)
    n_out = sum(1 for t in merged if t.get("out"))
    n_fc = sum(len(v) for v in fc.values())
    print(f"저장 완료: 총 {len(merged)}건 ({summarize(merged)}), 이상치 후보 {n_out}건, 예측 {n_fc}개 조합 -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
