import calendar
import datetime as dt
import hashlib
import io
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit(
        "필수 패키지 streamlit이 없습니다. 설치: pip install streamlit yfinance pandas numpy"
    ) from exc

try:
    import yfinance as yf
except ImportError:
    st.error("필수 패키지 yfinance가 없습니다. 설치: pip install yfinance")
    st.stop()

try:
    import pandas_market_calendars as mcal
except ImportError:
    mcal = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


DAY_NAME_TO_INT = {
    "MON": 0,
    "TUE": 1,
    "WED": 2,
    "THU": 3,
    "FRI": 4,
    "SAT": 5,
    "SUN": 6,
}

INT_TO_FREQ = {
    0: "W-MON",
    1: "W-TUE",
    2: "W-WED",
    3: "W-THU",
    4: "W-FRI",
    5: "W-SAT",
    6: "W-SUN",
}

FIXED_START_DATE = dt.date(2025, 12, 29)

@dataclass
class AssetConfig:
    ticker: str
    name: str
    weight: float
    day_of_week: str
    weekly_amount_base: float = 0.0
    target_amount_base: float = 0.0
    target_shares: float = 0.0


DEFAULT_ASSETS = [
    AssetConfig("VOO", "S&P500", 0.30, "월", 7500.0, 0.0, 0.0),
    AssetConfig("TLT", "20년+ 국채", 0.40, "화", 10000.0, 0.0, 0.0),
    AssetConfig("IEF", "7-10년 국채", 0.15, "수", 3500.0, 0.0, 0.0),
    AssetConfig("GLD", "금", 0.075, "목", 2000.0, 0.0, 0.0),
    AssetConfig("XLE", "에너지 섹터", 0.075, "금", 2000.0, 0.0, 0.0),
]

MANUAL_TX_COLUMNS = [
    "date",
    "ticker",
    "name",
    "shares",
    "price_usd",
    "price_krw",
    "amount_usd",
    "amount_base",
    "fx_rate",
]

ASSET_SCHEMA_VERSION = 3


@st.cache_data(ttl=3600)
def download_prices(tickers: List[str], start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date + dt.timedelta(days=1),
        auto_adjust=True,
        progress=False,
    )
    if data is None or data.empty:
        fallback_start = start_date - dt.timedelta(days=30)
        data = yf.download(
            tickers,
            start=fallback_start,
            end=end_date + dt.timedelta(days=1),
            auto_adjust=True,
            progress=False,
        )
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame) and "Close" in data.columns:
        prices = data["Close"]
    elif isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        prices = data["Adj Close"]
    else:
        prices = data

    if isinstance(prices, pd.Series):
        ticker = tickers[0]
        prices = prices.to_frame(name=ticker)

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices


def fetch_latest_prices(tickers: List[str]) -> pd.Series:
    if not tickers:
        return pd.Series(dtype=float)
    try:
        data = yf.download(
            tickers,
            period="1d",
            interval="1m",
            prepost=True,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception:
        return pd.Series(dtype=float)
    if data is None or data.empty:
        return pd.Series(dtype=float)

    latest: Dict[str, float] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker in data.columns.get_level_values(0):
                series = None
                if (ticker, "Close") in data.columns:
                    series = data[(ticker, "Close")]
                elif (ticker, "Adj Close") in data.columns:
                    series = data[(ticker, "Adj Close")]
                if series is None:
                    continue
                series = series.dropna()
                if not series.empty:
                    latest[ticker] = float(series.iloc[-1])
    else:
        series = None
        if "Close" in data.columns:
            series = data["Close"]
        elif "Adj Close" in data.columns:
            series = data["Adj Close"]
        if series is not None:
            series = series.dropna()
            if not series.empty:
                latest[tickers[0]] = float(series.iloc[-1])
    return pd.Series(latest, dtype=float)


@st.cache_data(ttl=3600)
def download_fx_rates(start_date: dt.date, end_date: dt.date) -> pd.Series:
    data = yf.download(
        "KRW=X",
        start=start_date,
        end=end_date + dt.timedelta(days=1),
        auto_adjust=True,
        progress=False,
    )
    if data is None or data.empty:
        fallback_start = start_date - dt.timedelta(days=30)
        data = yf.download(
            "KRW=X",
            start=fallback_start,
            end=end_date + dt.timedelta(days=1),
            auto_adjust=True,
            progress=False,
        )
    if data is None:
        return pd.Series(dtype=float)
    if isinstance(data, pd.DataFrame) and "Close" in data.columns:
        rates = data["Close"]
    elif isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        rates = data["Adj Close"]
    else:
        rates = data

    if isinstance(rates, pd.DataFrame):
        if "KRW=X" in rates.columns:
            rates = rates["KRW=X"]
        elif rates.shape[1] == 1:
            rates = rates.iloc[:, 0]
        else:
            rates = rates.iloc[:, 0]

    rates.index = pd.to_datetime(rates.index).tz_localize(None)
    return rates


def normalize_day_name(value: str) -> str:
    if not value:
        return "Mon"
    raw = str(value).strip()
    if not raw:
        return "Mon"
    kor_map = {
        "월": "Mon",
        "월요일": "Mon",
        "화": "Tue",
        "화요일": "Tue",
        "수": "Wed",
        "수요일": "Wed",
        "목": "Thu",
        "목요일": "Thu",
        "금": "Fri",
        "금요일": "Fri",
        "토": "Sat",
        "토요일": "Sat",
        "일": "Sun",
        "일요일": "Sun",
    }
    if raw in kor_map:
        return kor_map[raw]
    if raw and raw[0] in kor_map:
        return kor_map[raw[0]]

    upper = raw.upper()
    eng_full = {
        "MONDAY": "Mon",
        "TUESDAY": "Tue",
        "WEDNESDAY": "Wed",
        "THURSDAY": "Thu",
        "FRIDAY": "Fri",
        "SATURDAY": "Sat",
        "SUNDAY": "Sun",
    }
    if upper in eng_full:
        return eng_full[upper]

    value3 = upper[:3]
    if value3 in DAY_NAME_TO_INT:
        return value3.capitalize()

    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx <= 6:
            return list(DAY_NAME_TO_INT.keys())[idx].capitalize()

    return "Mon"


def canonical_day_label(value: object) -> str:
    kor_days = {
        "Mon": "월",
        "Tue": "화",
        "Wed": "수",
        "Thu": "목",
        "Fri": "금",
        "Sat": "토",
        "Sun": "일",
    }
    normalized = normalize_day_name(str(value))
    return kor_days.get(normalized, "월")


def cycle_weekdays(count: int) -> List[str]:
    cycle = ["월", "화", "수", "목", "금", "토", "일"]
    return [cycle[i % len(cycle)] for i in range(count)]


def coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_manual_tx_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=MANUAL_TX_COLUMNS)
    data = df.copy()
    for col in MANUAL_TX_COLUMNS:
        if col not in data.columns:
            data[col] = np.nan
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["name"] = data["name"].astype(str).str.strip()
    numeric_cols = [
        "shares",
        "price_usd",
        "price_krw",
        "amount_usd",
        "amount_base",
        "fx_rate",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data[MANUAL_TX_COLUMNS]


def autofill_manual_tx_df(
    df: pd.DataFrame | None,
    fx_rates: pd.Series | None,
    fx_mode: str,
    fx_lock_rate: float,
    base_currency: str,
    force_fx_refresh: bool = False,
) -> pd.DataFrame:
    data = normalize_manual_tx_df(df)
    if data.empty:
        return data
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for idx, row in data.iterrows():
        date_ts = row.get("date")
        if pd.isna(date_ts):
            continue
        date = date_ts.date()
        price_usd = coerce_float(row.get("price_usd"), np.nan)
        price_krw = coerce_float(row.get("price_krw"), np.nan)
        fx_rate = coerce_float(row.get("fx_rate"), np.nan)
        amount_base = coerce_float(row.get("amount_base"), np.nan)
        shares = coerce_float(row.get("shares"), np.nan)

        needs_fx = base_currency == "KRW" or (not pd.isna(price_krw) and price_krw > 0)
        if needs_fx and (force_fx_refresh or pd.isna(fx_rate) or fx_rate <= 0):
            if fx_mode == "custom" and not pd.isna(fx_lock_rate):
                fx_rate = fx_lock_rate
            elif fx_rates is not None and not fx_rates.empty:
                fx_rate = get_fx_rate_at_date(fx_rates, date)
            elif not pd.isna(fx_lock_rate):
                fx_rate = fx_lock_rate

        if not pd.isna(price_usd) and price_usd > 0:
            if not pd.isna(fx_rate) and fx_rate > 0 and (pd.isna(price_krw) or price_krw <= 0 or force_fx_refresh):
                price_krw = price_usd * fx_rate
        if pd.isna(price_usd) and not pd.isna(price_krw) and price_krw > 0:
            if not pd.isna(fx_rate) and fx_rate > 0:
                price_usd = price_krw / fx_rate

        if (pd.isna(shares) or shares <= 0) and not pd.isna(amount_base) and amount_base > 0:
            if base_currency == "KRW":
                if not pd.isna(price_krw) and price_krw > 0:
                    shares = amount_base / price_krw
                elif (
                    not pd.isna(price_usd)
                    and price_usd > 0
                    and not pd.isna(fx_rate)
                    and fx_rate > 0
                ):
                    shares = (amount_base / fx_rate) / price_usd
            else:
                if not pd.isna(price_usd) and price_usd > 0:
                    shares = amount_base / price_usd
                elif (
                    not pd.isna(price_krw)
                    and price_krw > 0
                    and not pd.isna(fx_rate)
                    and fx_rate > 0
                ):
                    shares = amount_base / (price_krw / fx_rate)

        data.at[idx, "price_usd"] = price_usd
        data.at[idx, "price_krw"] = price_krw
        data.at[idx, "fx_rate"] = fx_rate
        data.at[idx, "shares"] = shares

    data["date"] = data["date"].dt.normalize()
    return normalize_manual_tx_df(data)


def normalize_entry_df(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    data = df.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    if "ticker" in data.columns:
        data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    numeric_cols = ["amount_base", "price_usd", "price_krw", "fx_rate", "shares"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").round(6)
    return data[columns]


def values_different(left: object, right: object) -> bool:
    if pd.isna(left) and pd.isna(right):
        return False
    try:
        left_f = float(left)
        right_f = float(right)
        if pd.isna(left_f) and pd.isna(right_f):
            return False
        return not np.isclose(left_f, right_f, rtol=1e-6, atol=1e-9)
    except (TypeError, ValueError):
        return left != right


def resolve_fx_rate(
    date: dt.date,
    fx_rates: pd.Series | None,
    fx_mode: str,
    fx_lock_rate: float,
    fallback: float,
) -> float:
    if fx_mode == "custom" and not pd.isna(fx_lock_rate) and fx_lock_rate > 0:
        return fx_lock_rate
    if fx_rates is not None and not fx_rates.empty:
        return get_fx_rate_at_date(fx_rates, date)
    if not pd.isna(fx_lock_rate) and fx_lock_rate > 0:
        return fx_lock_rate
    return fallback


def resolve_price_for_base(
    price_usd: float,
    price_krw: float,
    fx_rate: float,
    base_currency: str,
) -> Tuple[float, float, float]:
    if base_currency == "KRW":
        if not pd.isna(price_krw) and price_krw > 0:
            return price_krw, price_usd, price_krw
        if (
            not pd.isna(price_usd)
            and price_usd > 0
            and not pd.isna(fx_rate)
            and fx_rate > 0
        ):
            price_krw = price_usd * fx_rate
            return price_krw, price_usd, price_krw
    else:
        if not pd.isna(price_usd) and price_usd > 0:
            return price_usd, price_usd, price_krw
        if (
            not pd.isna(price_krw)
            and price_krw > 0
            and not pd.isna(fx_rate)
            and fx_rate > 0
        ):
            price_usd = price_krw / fx_rate
            return price_usd, price_usd, price_krw
    return np.nan, price_usd, price_krw


def recalc_entry_df(
    current_df: pd.DataFrame,
    prev_df: pd.DataFrame | None,
    fx_rates: pd.Series | None,
    fx_mode: str,
    fx_lock_rate: float,
    base_currency: str,
    force_fx_refresh: bool,
    entry_columns: List[str],
) -> pd.DataFrame:
    current = normalize_entry_df(current_df, entry_columns)
    prev = normalize_entry_df(prev_df, entry_columns) if prev_df is not None else current.copy()
    rows = []
    for idx, row in current.iterrows():
        prev_row = prev.iloc[idx] if idx < len(prev) else pd.Series(dtype=object)
        date_ts = row.get("date")
        date = pd.to_datetime(date_ts, errors="coerce")
        date_value = date.date() if not pd.isna(date) else dt.date.today()

        amount_base = coerce_float(row.get("amount_base"), np.nan)
        shares = coerce_float(row.get("shares"), np.nan)
        price_usd = coerce_float(row.get("price_usd"), np.nan)
        price_krw = coerce_float(row.get("price_krw"), np.nan)
        fx_rate = coerce_float(row.get("fx_rate"), np.nan)

        changed_amount = values_different(prev_row.get("amount_base"), amount_base)
        changed_shares = values_different(prev_row.get("shares"), shares)
        changed_price_usd = values_different(prev_row.get("price_usd"), price_usd)
        changed_price_krw = values_different(prev_row.get("price_krw"), price_krw)
        changed_fx = values_different(prev_row.get("fx_rate"), fx_rate)

        needs_fx = base_currency == "KRW" or (not pd.isna(price_krw) and price_krw > 0)
        if needs_fx and (force_fx_refresh or changed_fx or pd.isna(fx_rate) or fx_rate <= 0):
            fx_rate = resolve_fx_rate(
                date_value, fx_rates, fx_mode, fx_lock_rate, fx_rate
            )

        if pd.isna(price_usd) and not pd.isna(price_krw) and price_krw > 0 and fx_rate > 0:
            price_usd = price_krw / fx_rate
        if pd.isna(price_krw) and not pd.isna(price_usd) and price_usd > 0 and fx_rate > 0:
            price_krw = price_usd * fx_rate
        if pd.isna(fx_rate) and not pd.isna(price_usd) and not pd.isna(price_krw) and price_usd > 0:
            fx_rate = price_krw / price_usd

        price_for_base, price_usd, price_krw = resolve_price_for_base(
            price_usd, price_krw, fx_rate, base_currency
        )

        if (
            (changed_amount and changed_shares)
            and not pd.isna(amount_base)
            and amount_base > 0
            and not pd.isna(shares)
            and shares > 0
            and pd.isna(price_for_base)
        ):
            price_for_base = amount_base / shares
            if base_currency == "KRW":
                price_krw = price_for_base
                if fx_rate > 0:
                    price_usd = price_krw / fx_rate
            else:
                price_usd = price_for_base
                if fx_rate > 0:
                    price_krw = price_usd * fx_rate

        if changed_shares and not changed_amount:
            if not pd.isna(shares) and shares > 0 and not pd.isna(price_for_base):
                amount_base = shares * price_for_base
            elif not pd.isna(amount_base) and amount_base > 0 and not pd.isna(shares) and shares > 0:
                price_for_base = amount_base / shares
        elif changed_amount and not changed_shares:
            if not pd.isna(amount_base) and amount_base > 0 and not pd.isna(price_for_base):
                shares = amount_base / price_for_base
            elif not pd.isna(amount_base) and amount_base > 0 and not pd.isna(shares) and shares > 0:
                price_for_base = amount_base / shares
        elif changed_amount and changed_shares:
            if not pd.isna(amount_base) and amount_base > 0 and not pd.isna(shares) and shares > 0:
                price_for_base = amount_base / shares
        elif changed_price_usd or changed_price_krw or force_fx_refresh:
            if not pd.isna(amount_base) and amount_base > 0 and not pd.isna(price_for_base):
                shares = amount_base / price_for_base
            elif not pd.isna(shares) and shares > 0 and not pd.isna(price_for_base):
                amount_base = shares * price_for_base
        else:
            if pd.isna(shares) and not pd.isna(amount_base) and amount_base > 0 and not pd.isna(price_for_base):
                shares = amount_base / price_for_base
            if pd.isna(amount_base) and not pd.isna(shares) and shares > 0 and not pd.isna(price_for_base):
                amount_base = shares * price_for_base

        if base_currency == "KRW" and not pd.isna(price_for_base) and price_for_base > 0:
            price_krw = price_for_base
            if pd.isna(price_usd) and not pd.isna(fx_rate) and fx_rate > 0:
                price_usd = price_krw / fx_rate
        if base_currency == "USD" and not pd.isna(price_for_base) and price_for_base > 0:
            price_usd = price_for_base
            if pd.isna(price_krw) and not pd.isna(fx_rate) and fx_rate > 0:
                price_krw = price_usd * fx_rate

        rows.append(
            {
                "date": row.get("date"),
                "ticker": row.get("ticker"),
                "name": row.get("name"),
                "amount_base": amount_base,
                "price_usd": price_usd,
                "price_krw": price_krw,
                "fx_rate": fx_rate,
                "shares": shares,
            }
        )
    return normalize_entry_df(pd.DataFrame(rows), entry_columns)


def get_manual_tx_path() -> Path:
    base_dir = Path(__file__).resolve().parent / "data"
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return Path(__file__).resolve().parent / "manual_transactions.csv"
    return base_dir / "manual_transactions.csv"


def get_manual_tx_backup_dir() -> Path:
    base_dir = Path(__file__).resolve().parent / "data" / "manual_tx_backups"
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return Path(__file__).resolve().parent
    return base_dir


def list_manual_tx_backups() -> List[Path]:
    backup_dir = get_manual_tx_backup_dir()
    if not backup_dir.exists():
        return []
    backups = sorted(backup_dir.glob("manual_tx_*.csv"), key=lambda p: p.name, reverse=True)
    return backups


def format_backup_label(path: Path) -> str:
    stem = path.stem
    if stem.startswith("manual_tx_"):
        raw = stem.replace("manual_tx_", "")
        try:
            parsed = dt.datetime.strptime(raw, "%Y%m%d_%H%M%S")
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return path.name
    return path.name


def manual_tx_digest(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    csv_text = df.to_csv(index=False)
    return hashlib.md5(csv_text.encode("utf-8")).hexdigest()


def load_manual_tx_file(path: Path) -> Tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return normalize_manual_tx_df(None), None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return normalize_manual_tx_df(None), f"직접 입력 내역 파일을 읽지 못했습니다: {exc}"
    return normalize_manual_tx_df(df), None


def save_manual_tx_file(df: pd.DataFrame, path: Path) -> str | None:
    try:
        df.to_csv(path, index=False)
    except Exception as exc:
        return f"직접 입력 내역 파일을 저장하지 못했습니다: {exc}"
    return None


def create_manual_tx_backup(df: pd.DataFrame) -> str | None:
    backup_dir = get_manual_tx_backup_dir()
    digest = manual_tx_digest(df)
    if not digest or digest == st.session_state.get("manual_tx_backup_digest"):
        return None
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"manual_tx_{timestamp}.csv"
    try:
        df.to_csv(backup_path, index=False)
    except Exception as exc:
        return f"직접 입력 내역 백업에 실패했습니다: {exc}"
    st.session_state["manual_tx_backup_digest"] = digest

    backups = list_manual_tx_backups()
    for old_path in backups[30:]:
        try:
            old_path.unlink()
        except OSError:
            continue
    return None


def persist_manual_tx_df(df: pd.DataFrame) -> str | None:
    path = get_manual_tx_path()
    if df is None or df.empty:
        if path.exists():
            try:
                path.unlink()
            except OSError as exc:
                return f"직접 입력 내역 파일을 삭제하지 못했습니다: {exc}"
        st.session_state["manual_tx_last_digest"] = ""
        return None
    digest = manual_tx_digest(df)
    if digest != st.session_state.get("manual_tx_last_digest"):
        error = save_manual_tx_file(df, path)
        if error is None:
            st.session_state["manual_tx_last_digest"] = digest
            backup_error = create_manual_tx_backup(df)
            if backup_error:
                return backup_error
        return error
    return None


def dataframe_to_pdf_bytes(title: str, df: pd.DataFrame) -> bytes | None:
    if not REPORTLAB_AVAILABLE:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    elements = [Paragraph(title, styles["Heading2"]), Spacer(1, 12)]
    safe_df = df.copy().astype(str)
    table_data = [safe_df.columns.tolist()] + safe_df.values.tolist()
    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()


def render_download_buttons(
    label: str,
    df: pd.DataFrame,
    filename_prefix: str,
    key_prefix: str,
) -> None:
    if df is None or df.empty:
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    col_csv, col_pdf = st.columns(2)
    col_csv.download_button(
        f"{label} CSV 다운로드",
        data=csv_bytes,
        file_name=f"{filename_prefix}.csv",
        mime="text/csv",
        key=f"{key_prefix}_csv",
    )
    if REPORTLAB_AVAILABLE:
        pdf_bytes = dataframe_to_pdf_bytes(label, df)
        col_pdf.download_button(
            f"{label} PDF 다운로드",
            data=pdf_bytes,
            file_name=f"{filename_prefix}.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_pdf",
        )
    else:
        col_pdf.caption("PDF 다운로드는 reportlab 설치가 필요합니다.")


def merge_manual_price_entries(
    existing_df: pd.DataFrame | None, entry_df: pd.DataFrame | None
) -> pd.DataFrame:
    existing = normalize_manual_tx_df(existing_df)
    if entry_df is None or entry_df.empty:
        return existing
    entries = []
    for _, row in entry_df.iterrows():
        date = pd.to_datetime(row.get("date"), errors="coerce")
        ticker = str(row.get("ticker", "")).strip().upper()
        if pd.isna(date) or not ticker:
            continue
        entries.append(
            {
                "date": date,
                "ticker": ticker,
                "name": str(row.get("name", "")).strip(),
                "shares": coerce_float(row.get("shares"), np.nan),
                "price_usd": coerce_float(row.get("price_usd"), np.nan),
                "price_krw": coerce_float(row.get("price_krw"), np.nan),
                "amount_usd": np.nan,
                "amount_base": coerce_float(row.get("amount_base"), np.nan),
                "fx_rate": coerce_float(row.get("fx_rate"), np.nan),
            }
        )
    if not entries:
        return existing
    new_df = pd.DataFrame(entries)
    existing["date"] = pd.to_datetime(existing["date"], errors="coerce").dt.normalize()
    existing["ticker"] = existing["ticker"].astype(str).str.upper().str.strip()
    new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce").dt.normalize()
    new_df["ticker"] = new_df["ticker"].astype(str).str.upper().str.strip()
    existing_index = existing.set_index(["date", "ticker"]).index
    new_index = new_df.set_index(["date", "ticker"]).index
    existing = existing[~existing_index.isin(new_index)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    return normalize_manual_tx_df(combined)


def parse_assets(df: pd.DataFrame) -> Tuple[List[AssetConfig], List[str]]:
    assets = []
    warnings = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            warnings.append("티커가 비어 있어 해당 자산을 건너뛰었습니다.")
            continue
        name = str(row.get("name", "")).strip() or ticker
        weight = coerce_float(row.get("weight", 0), 0.0)
        if weight == 0.0 and str(row.get("weight", "")).strip() not in ("", "0", "0.0"):
            warnings.append(f"{ticker} 비중이 올바르지 않아 0으로 설정했습니다.")
        day_of_week = normalize_day_name(str(row.get("day_of_week", "Mon")))
        weekly_amount_base = coerce_float(row.get("weekly_amount_base", 0), 0.0)
        target_amount_base = coerce_float(row.get("target_amount_base", 0), 0.0)
        target_shares = coerce_float(row.get("target_shares", 0), 0.0)
        assets.append(
            AssetConfig(
                ticker,
                name,
                weight,
                day_of_week,
                weekly_amount_base,
                target_amount_base,
                target_shares,
            )
        )
    return assets, warnings


def normalize_weights(assets: List[AssetConfig]) -> List[AssetConfig]:
    total = sum(a.weight for a in assets)
    if total <= 0:
        return assets
    return [
        AssetConfig(
            a.ticker,
            a.name,
            a.weight / total,
            a.day_of_week,
            a.weekly_amount_base,
            a.target_amount_base,
            a.target_shares,
        )
        for a in assets
    ]


def weekday_frequency(day_name: str) -> str:
    normalized = normalize_day_name(str(day_name))
    day_key = normalized.strip().upper()[:3]
    weekday = DAY_NAME_TO_INT.get(day_key, 0)
    return INT_TO_FREQ[weekday]


def generate_schedule(
    start_date: dt.date,
    end_date: dt.date,
    day_name: str,
    trading_days: set | None = None,
) -> List[dt.date]:
    freq = weekday_frequency(day_name)
    dates = [d.date() for d in pd.date_range(start_date, end_date, freq=freq)]
    if trading_days is None:
        return dates
    trading_days_sorted = sorted(trading_days)
    shifted_dates: List[dt.date] = []
    for date in dates:
        if date in trading_days:
            shifted = date
        else:
            idx = bisect_left(trading_days_sorted, date)
            if idx >= len(trading_days_sorted):
                continue
            shifted = trading_days_sorted[idx]
        if shifted > end_date:
            continue
        shifted_dates.append(shifted)

    unique_dates = []
    seen = set()
    for date in shifted_dates:
        if date in seen:
            continue
        seen.add(date)
        unique_dates.append(date)
    return unique_dates


def count_dates_by_month(dates: List[dt.date]) -> Dict[Tuple[int, int], int]:
    counts: Dict[Tuple[int, int], int] = {}
    for date in dates:
        key = (date.year, date.month)
        counts[key] = counts.get(key, 0) + 1
    return counts


def monthly_budget_from_weekly(
    assets: List[AssetConfig],
    year: int,
    month: int,
    trading_days: set | None = None,
) -> float:
    last_day = calendar.monthrange(year, month)[1]
    month_start = dt.date(year, month, 1)
    month_end = dt.date(year, month, last_day)
    total = 0.0
    for asset in assets:
        if asset.weekly_amount_base <= 0:
            continue
        schedule = generate_schedule(month_start, month_end, asset.day_of_week, trading_days)
        total += asset.weekly_amount_base * len(schedule)
    return total


def get_fx_rate_at_date(fx_rates: pd.Series, target_date: dt.date) -> float:
    if fx_rates is None or fx_rates.empty:
        return np.nan
    ts = pd.Timestamp(target_date)
    if ts in fx_rates.index:
        rate = fx_rates.loc[ts]
    else:
        prior = fx_rates.loc[:ts]
        if not prior.empty:
            rate = prior.iloc[-1]
        else:
            rate = fx_rates.iloc[0]
    if isinstance(rate, pd.Series):
        rate = rate.iloc[0]
    return float(rate) if not pd.isna(rate) else np.nan


def get_us_trading_days(start_date: dt.date, end_date: dt.date) -> set:
    if mcal is None:
        return set()
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date, end_date)
    sessions = schedule.index
    if getattr(sessions, "tz", None) is not None:
        sessions = sessions.tz_convert(None)
    return set(sessions.date)


def resolve_fx_lock_rate(
    fx_rates: pd.Series,
    fx_mode: str,
    custom_rate: float,
    start_date: dt.date,
) -> float:
    if fx_mode == "live":
        return np.nan
    if fx_mode == "custom":
        return custom_rate if custom_rate > 0 else np.nan
    return get_fx_rate_at_date(fx_rates, start_date)


def build_transactions(
    assets: List[AssetConfig],
    start_date: dt.date,
    end_date: dt.date,
    monthly_contribution: float,
    prices_usd: pd.DataFrame,
    fx_rates: pd.Series,
    base_currency: str,
    plan_mode: str,
    trading_days: set | None,
    fx_mode: str,
    fx_lock_rate: float,
) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    rows = []
    daily_index = pd.date_range(start_date, end_date, freq="D")
    prices_daily = prices_usd.reindex(daily_index).ffill()
    fx_daily = fx_rates.reindex(daily_index).ffill() if fx_rates is not None else None
    if isinstance(fx_daily, pd.DataFrame):
        if fx_daily.shape[1] == 1:
            fx_daily = fx_daily.iloc[:, 0]
        else:
            fx_daily = fx_daily.iloc[:, 0]

    for asset in assets:
        if asset.weight <= 0:
            continue
        schedule = generate_schedule(start_date, end_date, asset.day_of_week, trading_days)
        if not schedule:
            warnings.append(f"{asset.ticker}의 매수 일정이 없습니다.")
            continue
        counts = count_dates_by_month(schedule) if plan_mode == "monthly_weight" else {}
        for purchase_date in schedule:
            if plan_mode == "fixed_weekly":
                per_buy_amount = asset.weekly_amount_base
                if per_buy_amount <= 0:
                    continue
            else:
                month_key = (purchase_date.year, purchase_date.month)
                monthly_bucket = monthly_contribution * asset.weight
                per_buy_amount = monthly_bucket / counts[month_key]
            price = prices_daily.loc[pd.Timestamp(purchase_date), asset.ticker]
            if pd.isna(price):
                warnings.append(
                    f"{purchase_date} {asset.ticker} 가격이 없어 건너뛰었습니다."
                )
                continue
            if base_currency == "KRW":
                if fx_mode == "live":
                    fx_rate = fx_daily.loc[pd.Timestamp(purchase_date)]
                else:
                    fx_rate = fx_lock_rate
                if isinstance(fx_rate, pd.Series):
                    fx_rate = fx_rate.iloc[0]
                if pd.isna(fx_rate):
                    warnings.append(
                        f"{purchase_date} 환율이 없어 {asset.ticker}를 건너뛰었습니다."
                    )
                    continue
                amount_usd = per_buy_amount / fx_rate
                amount_base = per_buy_amount
            else:
                fx_rate = np.nan
                amount_usd = per_buy_amount
                amount_base = per_buy_amount

            shares = amount_usd / price
            rows.append(
                {
                    "date": pd.Timestamp(purchase_date),
                    "ticker": asset.ticker,
                    "name": asset.name,
                    "day_of_week": asset.day_of_week,
                    "weight": asset.weight,
                    "amount_base": amount_base,
                    "amount_usd": amount_usd,
                    "fx_rate": fx_rate,
                    "price_usd": price,
                    "shares": shares,
                }
            )

    transactions = pd.DataFrame(rows)
    if not transactions.empty:
        transactions = transactions.sort_values(["date", "ticker"]).reset_index(drop=True)
    return transactions, warnings


def positions_from_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame(columns=["ticker", "name", "shares", "cost_base", "cost_usd"])
    grouped = (
        transactions.groupby("ticker")
        .agg(
            name=("name", "first"),
            shares=("shares", "sum"),
            cost_base=("amount_base", "sum"),
            cost_usd=("amount_usd", "sum"),
        )
        .reset_index()
    )
    return grouped


def combine_positions(frames: List[pd.DataFrame]) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame(columns=["ticker", "name", "shares", "cost_base", "cost_usd"])
    combined = pd.concat(valid_frames, ignore_index=True)
    combined["cost_base"] = pd.to_numeric(combined.get("cost_base"), errors="coerce")
    combined["cost_usd"] = pd.to_numeric(combined.get("cost_usd"), errors="coerce")
    grouped = (
        combined.groupby("ticker", as_index=False)
        .agg(
            name=("name", "first"),
            shares=("shares", "sum"),
            cost_base=("cost_base", "sum"),
            cost_usd=("cost_usd", lambda x: x.sum(min_count=1)),
        )
        .reset_index(drop=True)
    )
    return grouped


def build_summary_from_positions(
    assets: List[AssetConfig],
    positions: pd.DataFrame,
    prices_usd: pd.DataFrame,
    fx_rate_for_value: float,
    base_currency: str,
) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "name",
                "target_weight",
                "target_amount_base",
                "target_shares",
                "shares",
                "avg_cost_base",
                "avg_cost_usd",
                "current_price_usd",
                "cost_base",
                "cost_usd",
                "market_value_base",
                "profit_base",
                "profit_pct",
                "progress_amount",
                "progress_shares",
                "remaining_amount",
                "remaining_shares",
                "actual_weight",
            ]
        )

    latest_prices = prices_usd.ffill().iloc[-1]
    asset_map = {asset.ticker: asset for asset in assets}
    summary_rows = []

    for _, row in positions.iterrows():
        ticker = row["ticker"]
        shares = row["shares"]
        cost_base = row.get("cost_base", np.nan)
        cost_usd = row.get("cost_usd", np.nan)
        avg_cost_base = (
            cost_base / shares if not pd.isna(shares) and shares > 0 and not pd.isna(cost_base) else np.nan
        )
        avg_cost_usd = (
            cost_usd / shares if not pd.isna(shares) and shares > 0 and not pd.isna(cost_usd) else np.nan
        )
        current_price = latest_prices.get(ticker, np.nan)
        market_value_usd = shares * current_price if not pd.isna(current_price) else np.nan
        if base_currency == "KRW":
            market_value_base = (
                market_value_usd * fx_rate_for_value if not pd.isna(market_value_usd) else np.nan
            )
        else:
            market_value_base = market_value_usd

        if not pd.isna(cost_base) and cost_base > 0 and not pd.isna(market_value_base):
            profit_base = market_value_base - cost_base
            profit_pct = profit_base / cost_base
        else:
            profit_base = np.nan
            profit_pct = np.nan

        asset = asset_map.get(ticker)
        target_weight = asset.weight if asset else np.nan
        target_amount_base = asset.target_amount_base if asset else 0.0
        target_shares = asset.target_shares if asset else 0.0
        progress_amount = (
            market_value_base / target_amount_base if target_amount_base else np.nan
        )
        progress_shares = shares / target_shares if target_shares else np.nan
        remaining_amount = (
            target_amount_base - market_value_base if target_amount_base else np.nan
        )
        remaining_shares = target_shares - shares if target_shares else np.nan

        summary_rows.append(
            {
                "ticker": ticker,
                "name": asset.name if asset else row.get("name", ticker),
                "target_weight": target_weight,
                "target_amount_base": target_amount_base,
                "target_shares": target_shares,
                "shares": shares,
                "avg_cost_base": avg_cost_base,
                "avg_cost_usd": avg_cost_usd,
                "current_price_usd": current_price,
                "cost_base": cost_base,
                "cost_usd": cost_usd,
                "market_value_base": market_value_base,
                "profit_base": profit_base,
                "profit_pct": profit_pct,
                "progress_amount": progress_amount,
                "progress_shares": progress_shares,
                "remaining_amount": remaining_amount,
                "remaining_shares": remaining_shares,
            }
        )

    summary = pd.DataFrame(summary_rows)
    total_value = summary["market_value_base"].sum()
    if total_value > 0:
        summary["actual_weight"] = summary["market_value_base"] / total_value
    else:
        summary["actual_weight"] = np.nan

    summary = summary.sort_values("target_weight", ascending=False).reset_index(drop=True)
    return summary


def build_summary(
    assets: List[AssetConfig],
    transactions: pd.DataFrame,
    prices_usd: pd.DataFrame,
    fx_rates: pd.Series,
    base_currency: str,
) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()

    latest_fx = fx_rates.ffill().iloc[-1] if fx_rates is not None else np.nan
    positions = positions_from_transactions(transactions)
    return build_summary_from_positions(assets, positions, prices_usd, latest_fx, base_currency)


def build_daily_timeseries(
    transactions: pd.DataFrame,
    prices_usd: pd.DataFrame,
    fx_rates: pd.Series,
    base_currency: str,
    fx_mode: str,
    fx_lock_rate: float,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()

    daily_index = pd.date_range(start_date, end_date, freq="D")
    prices_daily = prices_usd.reindex(daily_index).ffill()
    if base_currency == "KRW":
        if fx_mode == "live":
            fx_daily = fx_rates.reindex(daily_index).ffill()
        else:
            fx_daily = pd.Series(fx_lock_rate, index=daily_index)
    else:
        fx_daily = pd.Series(1.0, index=daily_index)

    portfolio_value = pd.Series(0.0, index=daily_index)

    for ticker in prices_daily.columns:
        tx = transactions[transactions["ticker"] == ticker]
        if tx.empty:
            continue
        shares_by_date = tx.groupby("date")["shares"].sum()
        shares_daily = shares_by_date.reindex(daily_index, fill_value=0).cumsum()
        value_usd = shares_daily * prices_daily[ticker]
        portfolio_value = portfolio_value.add(value_usd, fill_value=0)

    portfolio_value_base = portfolio_value * fx_daily

    invested = (
        transactions.groupby("date")["amount_base"].sum().reindex(daily_index, fill_value=0).cumsum()
    )

    return pd.DataFrame(
        {
            "portfolio_value_base": portfolio_value_base,
            "invested_base": invested,
            "profit_base": portfolio_value_base - invested,
        }
    )


def build_upcoming_schedule(
    assets: List[AssetConfig],
    start_date: dt.date,
    end_date: dt.date,
    monthly_contribution: float,
    plan_mode: str,
    trading_days: set | None,
) -> pd.DataFrame:
    rows = []
    current = dt.date(start_date.year, start_date.month, 1)
    last = dt.date(end_date.year, end_date.month, 1)

    while current <= last:
        month_start = current
        next_month = (month_start.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        month_end = next_month - dt.timedelta(days=1)
        for asset in assets:
            if asset.weight <= 0:
                continue
            schedule = generate_schedule(
                month_start, month_end, asset.day_of_week, trading_days
            )
            if not schedule:
                continue
            if plan_mode == "fixed_weekly":
                per_buy_amount = asset.weekly_amount_base
                if per_buy_amount <= 0:
                    continue
            else:
                per_buy_amount = (monthly_contribution * asset.weight) / len(schedule)
            for purchase_date in schedule:
                if start_date <= purchase_date <= end_date:
                    rows.append(
                        {
                            "date": pd.Timestamp(purchase_date),
                            "ticker": asset.ticker,
                            "name": asset.name,
                            "planned_amount_base": per_buy_amount,
                        }
                    )
        current = next_month

    schedule_df = pd.DataFrame(rows)
    if not schedule_df.empty:
        schedule_df = schedule_df.sort_values("date").reset_index(drop=True)
    return schedule_df


def build_planned_amount_map(
    assets: List[AssetConfig],
    start_date: dt.date,
    end_date: dt.date,
    monthly_contribution: float,
    plan_mode: str,
    trading_days: set | None,
) -> Dict[Tuple[dt.date, str], float]:
    schedule_df = build_upcoming_schedule(
        assets, start_date, end_date, monthly_contribution, plan_mode, trading_days
    )
    planned: Dict[Tuple[dt.date, str], float] = {}
    if schedule_df.empty:
        return planned
    for row in schedule_df.itertuples(index=False):
        planned[(row.date.date(), row.ticker)] = row.planned_amount_base
    return planned


def build_period_report(transactions: pd.DataFrame, period: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if transactions.empty:
        return pd.DataFrame(), pd.DataFrame()
    tx = transactions.copy()
    if period == "daily":
        tx["period"] = tx["date"].dt.date
    elif period == "weekly":
        tx["period"] = tx["date"].dt.to_period("W-MON").apply(lambda p: p.start_time.date())
    else:
        tx["period"] = tx["date"].dt.to_period("M").apply(lambda p: p.start_time.date())

    totals = (
        tx.groupby("period")
        .agg(
            buys=("ticker", "count"),
            invested_base=("amount_base", "sum"),
            shares=("shares", "sum"),
        )
        .reset_index()
    )
    by_asset = (
        tx.groupby(["period", "ticker"])
        .agg(
            buys=("ticker", "count"),
            invested_base=("amount_base", "sum"),
            shares=("shares", "sum"),
        )
        .reset_index()
    )
    return totals, by_asset


def extract_holdings_tickers(holdings_bytes: bytes) -> List[str]:
    if not holdings_bytes:
        return []
    try:
        df = pd.read_csv(io.BytesIO(holdings_bytes))
    except Exception:
        return []
    if "ticker" not in df.columns:
        return []
    tickers = df["ticker"].astype(str).str.upper().str.strip()
    return sorted({ticker for ticker in tickers if ticker})


def extract_manual_tickers(df: pd.DataFrame | None) -> List[str]:
    if df is None or df.empty or "ticker" not in df.columns:
        return []
    tickers = df["ticker"].astype(str).str.upper().str.strip()
    return sorted({ticker for ticker in tickers if ticker})


def parse_actual_holdings(
    uploaded_file: object,
    base_currency: str,
    fx_rate_for_cost: float,
) -> Tuple[pd.DataFrame, List[str]]:
    if uploaded_file is None:
        return pd.DataFrame(), []
    warnings: List[str] = []
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        warnings.append(f"CSV를 읽을 수 없습니다: {exc}")
        return pd.DataFrame(), warnings

    if "ticker" not in df.columns or "shares" not in df.columns:
        warnings.append("CSV에 필요한 컬럼: ticker, shares")
        return pd.DataFrame(), warnings

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df[df["ticker"] != ""]
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df = df[df["shares"].notna() & (df["shares"] > 0)]
    if df.empty:
        warnings.append("보유 CSV에서 유효한 행을 찾지 못했습니다.")
        return pd.DataFrame(), warnings

    if "cost_base" not in df.columns and "avg_cost_base" not in df.columns:
        warnings.append("보유 CSV에 cost_base가 없어 손익 지표가 낮게 계산될 수 있습니다.")

    cost_base = (
        pd.to_numeric(df["cost_base"], errors="coerce")
        if "cost_base" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    avg_cost_base = (
        pd.to_numeric(df["avg_cost_base"], errors="coerce")
        if "avg_cost_base" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    if not avg_cost_base.isna().all():
        cost_base = cost_base.fillna(avg_cost_base * df["shares"])
    cost_base = cost_base.fillna(0.0)

    cost_usd = (
        pd.to_numeric(df["cost_usd"], errors="coerce")
        if "cost_usd" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    avg_cost_usd = (
        pd.to_numeric(df["avg_cost_usd"], errors="coerce")
        if "avg_cost_usd" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    if not avg_cost_usd.isna().all():
        cost_usd = cost_usd.fillna(avg_cost_usd * df["shares"])

    if cost_usd.isna().all():
        if base_currency == "USD":
            cost_usd = cost_base
        elif base_currency == "KRW" and fx_rate_for_cost and not pd.isna(fx_rate_for_cost):
            cost_usd = cost_base / fx_rate_for_cost
            warnings.append("보유 CSV의 cost_usd를 현재 환율로 추정했습니다.")

    name = df.get("name")
    if name is None:
        name = df["ticker"]

    parsed = pd.DataFrame(
        {
            "ticker": df["ticker"],
            "name": name,
            "shares": df["shares"],
            "cost_base": cost_base,
            "cost_usd": cost_usd,
        }
    )
    return parsed, warnings


def parse_manual_transactions(
    df: pd.DataFrame,
    base_currency: str,
    fx_rates: pd.Series | None,
    fx_mode: str,
    fx_lock_rate: float,
    planned_amounts: Dict[Tuple[dt.date, str], float] | None = None,
    asset_name_map: Dict[str, str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    if df is None or df.empty:
        return pd.DataFrame(), []
    warnings: List[str] = []
    rows = []
    data = normalize_manual_tx_df(df)
    for idx, row in data.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            warnings.append(f"{idx + 1}행: 티커가 비어 있습니다.")
            continue
        raw_date = row.get("date")
        date_ts = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(date_ts):
            warnings.append(f"{idx + 1}행: 날짜가 올바르지 않습니다.")
            continue
        date = date_ts.date()

        name = str(row.get("name", "")).strip()
        if not name and asset_name_map:
            name = asset_name_map.get(ticker, "")
        if not name:
            name = ticker

        shares = coerce_float(row.get("shares"), np.nan)
        price_usd = coerce_float(row.get("price_usd"), np.nan)
        price_krw = coerce_float(row.get("price_krw"), np.nan)
        amount_usd = coerce_float(row.get("amount_usd"), np.nan)
        amount_base = coerce_float(row.get("amount_base"), np.nan)
        fx_rate = coerce_float(row.get("fx_rate"), np.nan)

        if (pd.isna(amount_base) or amount_base <= 0) and planned_amounts:
            amount_base = planned_amounts.get((date, ticker), np.nan)
            if pd.isna(amount_base):
                warnings.append(
                    f"{idx + 1}행: 계획된 매수금액을 찾을 수 없어 제외했습니다."
                )
                continue

        needs_fx = base_currency == "KRW" or (not pd.isna(price_krw) and price_krw > 0)
        if needs_fx and (pd.isna(fx_rate) or fx_rate <= 0):
            if fx_mode == "live" and fx_rates is not None and not fx_rates.empty:
                fx_rate = get_fx_rate_at_date(fx_rates, date)
            else:
                fx_rate = fx_lock_rate
        if base_currency == "USD" and not needs_fx:
            fx_rate = np.nan

        if pd.isna(amount_usd) or amount_usd <= 0:
            if not pd.isna(price_usd) and price_usd > 0 and not pd.isna(shares) and shares > 0:
                amount_usd = price_usd * shares
            elif (
                base_currency == "USD"
                and not pd.isna(amount_base)
                and amount_base > 0
            ):
                amount_usd = amount_base
            elif (
                not pd.isna(amount_base)
                and amount_base > 0
                and not pd.isna(fx_rate)
                and fx_rate > 0
            ):
                amount_usd = amount_base / fx_rate

        if pd.isna(shares) or shares <= 0:
            if not pd.isna(amount_usd) and amount_usd > 0 and not pd.isna(price_usd) and price_usd > 0:
                shares = amount_usd / price_usd
            elif (
                not pd.isna(amount_base)
                and amount_base > 0
                and not pd.isna(price_krw)
                and price_krw > 0
                and base_currency == "KRW"
            ):
                shares = amount_base / price_krw

        if pd.isna(amount_base) or amount_base <= 0:
            if base_currency == "USD":
                if not pd.isna(amount_usd) and amount_usd > 0:
                    amount_base = amount_usd
            elif not pd.isna(amount_usd) and amount_usd > 0 and not pd.isna(fx_rate) and fx_rate > 0:
                amount_base = amount_usd * fx_rate

        if pd.isna(price_usd) or price_usd <= 0:
            if not pd.isna(price_krw) and price_krw > 0 and not pd.isna(fx_rate) and fx_rate > 0:
                price_usd = price_krw / fx_rate
            elif not pd.isna(amount_usd) and amount_usd > 0 and not pd.isna(shares) and shares > 0:
                price_usd = amount_usd / shares
        if pd.isna(price_krw) or price_krw <= 0:
            if not pd.isna(price_usd) and price_usd > 0 and not pd.isna(fx_rate) and fx_rate > 0:
                price_krw = price_usd * fx_rate

        if pd.isna(shares) or shares <= 0:
            warnings.append(f"{idx + 1}행: 수량을 계산할 수 없어 제외했습니다.")
            continue
        if pd.isna(amount_usd) or amount_usd <= 0:
            warnings.append(f"{idx + 1}행: 매수금(USD)을 계산할 수 없어 제외했습니다.")
            continue
        if pd.isna(amount_base) or amount_base <= 0:
            warnings.append(
                f"{idx + 1}행: 매수금({base_currency})을 계산할 수 없어 제외했습니다."
            )
            continue

        rows.append(
            {
                "date": pd.Timestamp(date),
                "ticker": ticker,
                "name": name,
                "shares": shares,
                "price_usd": price_usd,
                "price_krw": price_krw,
                "amount_usd": amount_usd,
                "amount_base": amount_base,
                "fx_rate": fx_rate,
            }
        )

    transactions = pd.DataFrame(rows)
    if not transactions.empty:
        transactions = transactions.sort_values(["date", "ticker"]).reset_index(drop=True)
    return transactions, warnings


def build_rebalance_plan(
    summary: pd.DataFrame,
    assets: List[AssetConfig],
    monthly_contribution: float,
    today: dt.date,
    trading_days: set | None,
) -> pd.DataFrame:
    if summary.empty or monthly_contribution <= 0:
        return pd.DataFrame()

    total_value = summary["market_value_base"].sum()
    target_total = total_value + monthly_contribution
    current_values = summary.set_index("ticker")["market_value_base"].to_dict()

    gaps: Dict[str, float] = {}
    for asset in assets:
        if asset.weight <= 0:
            continue
        current_value = current_values.get(asset.ticker, 0.0)
        target_value = target_total * asset.weight
        gaps[asset.ticker] = target_value - current_value

    positive_gaps = {k: max(v, 0.0) for k, v in gaps.items()}
    total_gap = sum(positive_gaps.values())

    next_month = (today.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    next_month_end = (next_month.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)

    rows = []
    for asset in assets:
        if asset.weight <= 0:
            continue
        if total_gap > 0:
            recommended = monthly_contribution * (positive_gaps.get(asset.ticker, 0.0) / total_gap)
        else:
            recommended = monthly_contribution * asset.weight
        schedule = generate_schedule(
            next_month, next_month_end, asset.day_of_week, trading_days
        )
        per_buy = recommended / len(schedule) if schedule else np.nan
        next_buy_candidates = generate_schedule(
            today, today + dt.timedelta(days=30), asset.day_of_week, trading_days
        )
        next_buy = next_buy_candidates[0] if next_buy_candidates else None

        rows.append(
            {
                "ticker": asset.ticker,
                "target_weight": asset.weight,
                "current_value": current_values.get(asset.ticker, 0.0),
                "target_value": target_total * asset.weight,
                "gap_to_target": gaps.get(asset.ticker, 0.0),
                "recommended_monthly_buy": recommended,
                "per_buy_amount": per_buy,
                "next_buy_date": next_buy,
                "next_month_buys": len(schedule),
            }
        )

    rebalance = pd.DataFrame(rows)
    return rebalance.sort_values("recommended_monthly_buy", ascending=False).reset_index(drop=True)


def next_deposit_date(today: dt.date, deposit_day: int) -> dt.date:
    last_day = calendar.monthrange(today.year, today.month)[1]
    day = min(deposit_day, last_day)
    candidate = dt.date(today.year, today.month, day)
    if candidate < today:
        next_month = (today.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        last_day = calendar.monthrange(next_month.year, next_month.month)[1]
        day = min(deposit_day, last_day)
        candidate = dt.date(next_month.year, next_month.month, day)
    return candidate


def next_weekday(from_date: dt.date, weekday: int) -> dt.date:
    days_ahead = (weekday - from_date.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return from_date + dt.timedelta(days=days_ahead)


def format_currency(value: float, currency: str) -> str:
    if pd.isna(value):
        return "-"
    if currency == "KRW":
        return f"{value:,.0f} KRW"
    return f"{value:,.2f} USD"


def format_unit_price(value: float, currency: str) -> str:
    if pd.isna(value):
        return "-"
    if currency == "KRW":
        return f"{value:,.0f} KRW"
    return f"{value:,.2f} USD"


def main() -> None:
    st.set_page_config(page_title="ETF 자동투자 대시보드", layout="wide")
    st.title("ETF 자동투자 대시보드")
    st.caption("직접 입력한 매수 내역을 기반으로 계산합니다.")


    today = dt.date.today()

    base_currency = "USD"
    fx_mode = "live"
    custom_fx_rate = np.nan

    st.sidebar.header("계획 설정")
    st.sidebar.date_input("시작일", value=FIXED_START_DATE, disabled=True)
    start_date = FIXED_START_DATE
    plan_choice = st.sidebar.selectbox(
        "매수 방식",
        ["월 납입액+비중", "주간 고정금액(종목별)"],
        index=1,
    )
    plan_mode = {
        "월 납입액+비중": "monthly_weight",
        "주간 고정금액(종목별)": "fixed_weekly",
    }[plan_choice]
    monthly_contribution = 0.0
    if plan_mode == "monthly_weight":
        monthly_contribution = st.sidebar.number_input(
            f"월 납입액({base_currency})", min_value=0.0, value=100000.0, step=10000.0
        )
    apply_us_holidays = st.sidebar.checkbox(
        "미국 휴장일 반영(휴장 시 다음 거래일로 이월)", value=True
    )
    st.sidebar.caption("기준 통화: USD (고정)")
    use_prepost = st.sidebar.checkbox("장외 가격 포함(새로고침 시)", value=True)
    if st.sidebar.button("데이터 새로고침"):
        st.session_state["refresh_nonce"] = st.session_state.get("refresh_nonce", 0) + 1
        st.cache_data.clear()
        st.rerun()

    use_manual_transactions = True

    st.sidebar.header("리마인더")
    deposit_day = st.sidebar.number_input(
        "월 납입일", min_value=1, max_value=28, value=1
    )
    reminder_days = st.sidebar.number_input(
        "리마인더 기간(일)", min_value=1, max_value=60, value=14
    )

    st.sidebar.header("자산")
    asset_columns = [
        "ticker",
        "name",
        "weight",
        "day_of_week",
        "weekly_amount_base",
        "target_amount_base",
        "target_shares",
    ]
    asset_column_config = {
        "ticker": "티커",
        "name": "이름",
        "weight": "비중",
        "day_of_week": st.column_config.SelectboxColumn(
            "요일",
            options=["월", "화", "수", "목", "금", "토", "일"],
            required=True,
        ),
        "weekly_amount_base": f"주간 금액({base_currency})",
        "target_amount_base": f"목표 금액({base_currency})",
        "target_shares": "목표 수량",
    }
    if "assets_df" not in st.session_state:
        st.session_state["assets_df"] = pd.DataFrame([a.__dict__ for a in DEFAULT_ASSETS])[
            asset_columns
        ]

    if st.session_state.get("assets_version", 0) < ASSET_SCHEMA_VERSION:
        assets_df_migrate = st.session_state["assets_df"].copy()
        for col in asset_columns:
            if col not in assets_df_migrate.columns:
                assets_df_migrate[col] = 0.0
        assets_df_migrate["ticker"] = (
            assets_df_migrate["ticker"].astype(str).str.upper().str.strip()
        )
        assets_df_migrate["day_of_week"] = assets_df_migrate["day_of_week"].map(
            canonical_day_label
        )
        if assets_df_migrate["day_of_week"].nunique() == 1 and len(assets_df_migrate) >= 5:
            default_day_map = {a.ticker: a.day_of_week for a in DEFAULT_ASSETS}
            mapped = assets_df_migrate["ticker"].map(default_day_map)
            if mapped.notna().any():
                assets_df_migrate["day_of_week"] = mapped.fillna(
                    assets_df_migrate["day_of_week"]
                )
            if assets_df_migrate["day_of_week"].nunique() == 1:
                assets_df_migrate["day_of_week"] = cycle_weekdays(len(assets_df_migrate))
            st.session_state["show_day_fix_notice"] = True

        st.session_state["assets_df"] = assets_df_migrate[asset_columns]
        st.session_state["assets_version"] = ASSET_SCHEMA_VERSION

    st.session_state["assets_df"]["day_of_week"] = st.session_state["assets_df"][
        "day_of_week"
    ].map(canonical_day_label)

    if st.sidebar.button("자산 설정 초기화"):
        st.session_state["assets_df"] = pd.DataFrame([a.__dict__ for a in DEFAULT_ASSETS])[
            asset_columns
        ]
        st.session_state["assets_df"]["day_of_week"] = st.session_state["assets_df"][
            "day_of_week"
        ].map(canonical_day_label)
        st.session_state["assets_version"] = ASSET_SCHEMA_VERSION
        st.session_state["show_day_fix_notice"] = True
        st.rerun()
    assets_df_sidebar = st.sidebar.data_editor(
        st.session_state["assets_df"],
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_config=asset_column_config,
        key="assets_editor_sidebar",
    )
    assets_df_sidebar["day_of_week"] = assets_df_sidebar["day_of_week"].map(
        canonical_day_label
    )
    st.session_state["assets_df"] = assets_df_sidebar

    st.subheader("자산 편집")
    st.caption("홈 화면에서 직접 수치를 수정할 수 있습니다.")
    assets_df_main = st.data_editor(
        st.session_state["assets_df"],
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_config=asset_column_config,
        key="assets_editor_main",
    )
    assets_df_main["day_of_week"] = assets_df_main["day_of_week"].map(canonical_day_label)
    st.session_state["assets_df"] = assets_df_main
    if "manual_tx_df" not in st.session_state:
        manual_path = get_manual_tx_path()
        loaded_manual, load_error = load_manual_tx_file(manual_path)
        st.session_state["manual_tx_df"] = loaded_manual
        st.session_state["manual_tx_last_digest"] = manual_tx_digest(loaded_manual)
        st.session_state["manual_tx_backup_digest"] = manual_tx_digest(loaded_manual)
        if load_error:
            st.warning(load_error)

    st.sidebar.header("실제 보유(선택)")
    holdings_file = st.sidebar.file_uploader("CSV 업로드", type=["csv"])
    include_actual = True
    if holdings_file is not None:
        include_actual = st.sidebar.checkbox("실제 보유 포함", value=True)
    holdings_bytes = holdings_file.getvalue() if holdings_file is not None else None

    if st.session_state.pop("show_day_fix_notice", False):
        st.info("요일 값이 모두 동일해 월~금으로 자동 배치했습니다. 자산 편집에서 확인하세요.")

    assets, asset_warnings = parse_assets(st.session_state["assets_df"])
    if asset_warnings:
        for message in asset_warnings:
            st.warning(message)
    asset_name_map = {asset.ticker: asset.name for asset in assets}

    normalize = st.sidebar.checkbox("비중 100%로 정규화", value=True)
    if normalize:
        assets = normalize_weights(assets)

    trading_days = None
    if apply_us_holidays:
        if mcal is None:
            st.error(
                "미국 휴장일 반영을 위해 pandas-market-calendars 설치가 필요합니다. "
                "설치: pip install pandas-market-calendars"
            )
            st.stop()
        schedule_start = min(start_date, today)
        reference_date = max(today, start_date)
        next_month_start = (reference_date.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        next_month_end = (next_month_start.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
        schedule_end = max(reference_date + dt.timedelta(days=60), next_month_end)
        trading_days = get_us_trading_days(schedule_start, schedule_end)
        if not trading_days:
            st.error("미국 거래일 캘린더를 가져오지 못했습니다.")
            st.stop()

    monthly_contribution_effective = monthly_contribution
    if plan_mode == "fixed_weekly":
        weekly_total = sum(a.weekly_amount_base for a in assets)
        month_ref = start_date if start_date > today else today
        next_month = (month_ref.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        monthly_contribution_effective = monthly_budget_from_weekly(
            assets, next_month.year, next_month.month, trading_days
        )
        st.sidebar.metric("주간 총액", format_currency(weekly_total, base_currency))
        st.sidebar.metric(
            "다음 달 예상 납입액", format_currency(monthly_contribution_effective, base_currency)
        )
        if weekly_total <= 0:
            st.error("주간 금액이 0입니다. 자산의 주간 금액을 입력하세요.")
            st.stop()

    weights_sum = sum(a.weight for a in assets)
    if not np.isclose(weights_sum, 1.0) and weights_sum > 0:
        st.warning("비중 합계가 100%가 아닙니다. 정규화를 고려하세요.")

    manual_df = st.session_state.get("manual_tx_df")
    has_manual_rows = (
        isinstance(manual_df, pd.DataFrame) and not manual_df.dropna(how="all").empty
    )
    if start_date > today:
        if has_manual_rows:
            st.info("매수 시작일은 미래지만 직접 입력 내역이 있어 계산에 반영합니다.")
        else:
            st.info("매수 시작일이 미래입니다. 현재는 거래 내역이 없습니다.")

    with st.expander("매수 내역 입력(직접)", expanded=use_manual_transactions):
        st.caption(
            "체결가(USD)와 계획 매수금액을 기준으로 수량/매수금을 자동 계산합니다."
        )
        entry_date = st.date_input(
            "체결가 입력일",
            value=today if start_date <= today else start_date,
            key="price_entry_date",
        )
        entry_trading_days = trading_days
        if apply_us_holidays and trading_days:
            trading_min = min(trading_days)
            trading_max = max(trading_days)
            if entry_date < trading_min or entry_date > trading_max:
                last_day = calendar.monthrange(entry_date.year, entry_date.month)[1]
                entry_month_start = dt.date(entry_date.year, entry_date.month, 1)
                entry_month_end = dt.date(entry_date.year, entry_date.month, last_day)
                entry_trading_days = get_us_trading_days(entry_month_start, entry_month_end)

        schedule_window_end = entry_date + dt.timedelta(days=7)
        planned_window = build_upcoming_schedule(
            assets,
            entry_date,
            schedule_window_end,
            monthly_contribution_effective,
            plan_mode,
            entry_trading_days,
        )
        if planned_window.empty:
            st.info("해당 기간에 예정된 매수가 없습니다. 아래 직접 입력 테이블을 사용하세요.")
        else:
            next_plan_date = planned_window["date"].min()
            planned_for_date = planned_window[planned_window["date"] == next_plan_date].copy()
            entry_df = planned_for_date.copy()
            entry_df["date"] = pd.to_datetime(entry_df["date"], errors="coerce").dt.normalize()
            if next_plan_date.date() != entry_date:
                st.info(
                    f"{entry_date}에는 예정 매수가 없어 다음 매수일 {next_plan_date.date()} 항목을 표시했습니다."
                )
            entry_df = entry_df.rename(columns={"planned_amount_base": "amount_base"})
            entry_df["price_usd"] = np.nan
            entry_df["shares"] = np.nan

            existing_manual = normalize_manual_tx_df(st.session_state.get("manual_tx_df"))
            if not existing_manual.empty:
                existing_manual = existing_manual.copy()
                existing_manual["date"] = pd.to_datetime(
                    existing_manual["date"], errors="coerce"
                ).dt.normalize()
                existing_manual["ticker"] = (
                    existing_manual["ticker"].astype(str).str.upper().str.strip()
                )
                entry_df = entry_df.merge(
                    existing_manual[
                        ["date", "ticker", "name", "price_usd", "amount_base", "shares"]
                    ],
                    on=["date", "ticker"],
                    how="left",
                    suffixes=("", "_existing"),
                )
                entry_df["price_usd"] = entry_df["price_usd_existing"].combine_first(
                    entry_df["price_usd"]
                )
                entry_df["amount_base"] = entry_df["amount_base_existing"].combine_first(
                    entry_df["amount_base"]
                )
                entry_df["shares"] = entry_df["shares_existing"].combine_first(
                    entry_df["shares"]
                )
                entry_df["name"] = entry_df["name_existing"].combine_first(entry_df["name"])
                entry_df = entry_df.drop(
                    columns=[
                        "price_usd_existing",
                        "amount_base_existing",
                        "shares_existing",
                        "name_existing",
                    ]
                )

            entry_columns = [
                "date",
                "ticker",
                "name",
                "amount_base",
                "price_usd",
                "shares",
            ]
            entry_column_config = {
                "date": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
                "ticker": "티커",
                "name": "이름",
                "amount_base": st.column_config.NumberColumn(
                    f"계획 매수금({base_currency})", format="%.2f"
                ),
                "price_usd": st.column_config.NumberColumn("체결가(USD)", format="%.4f"),
                "shares": st.column_config.NumberColumn("매수 수량", format="%.6f"),
            }
            if st.session_state.get("price_entry_seed") != entry_date:
                st.session_state["price_entry_seed"] = entry_date
                st.session_state["price_entry_df"] = entry_df[entry_columns]
                st.session_state.pop("price_entry_editor", None)

            prev_entry_df = st.session_state.get("price_entry_df")
            price_entry_df = st.data_editor(
                st.session_state["price_entry_df"],
                num_rows="fixed",
                use_container_width=True,
                hide_index=True,
                column_config=entry_column_config,
                disabled=["date", "ticker", "name"],
                key="price_entry_editor",
            )
            entry_fx_rates = None
            entry_fx_lock_rate = np.nan
            refresh_nonce = st.session_state.get("refresh_nonce", 0)
            force_fx_refresh = (
                st.session_state.get("price_entry_refresh_nonce") != refresh_nonce
            )
            recalculated_df = recalc_entry_df(
                price_entry_df,
                prev_entry_df,
                entry_fx_rates,
                fx_mode,
                entry_fx_lock_rate,
                base_currency,
                force_fx_refresh,
                entry_columns,
            )
            price_entry_df = normalize_entry_df(price_entry_df, entry_columns)
            if not recalculated_df.equals(price_entry_df):
                st.session_state["price_entry_refresh_nonce"] = refresh_nonce
                st.session_state["price_entry_df"] = recalculated_df
                st.rerun()
            st.session_state["price_entry_df"] = price_entry_df
            st.session_state["price_entry_refresh_nonce"] = refresh_nonce

            if st.button("입력 내역 반영"):
                st.session_state["manual_tx_df"] = merge_manual_price_entries(
                    st.session_state.get("manual_tx_df"), price_entry_df
                )
                st.session_state["manual_tx_df"] = normalize_manual_tx_df(
                    st.session_state["manual_tx_df"]
                )
                st.success("직접 입력 내역에 반영했습니다.")

        st.divider()
        st.caption("직접 입력(고급): 날짜/티커를 직접 추가하거나 계획에 없는 거래를 기록할 수 있습니다.")
        manual_column_config = {
            "date": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
            "ticker": "티커",
            "name": "이름",
            "amount_base": st.column_config.NumberColumn(
                f"매수금({base_currency})", format="%.2f"
            ),
            "price_usd": st.column_config.NumberColumn("체결가(USD)", format="%.4f"),
            "shares": st.column_config.NumberColumn("매수 수량", format="%.6f"),
        }

        manual_upload = st.file_uploader("매수 내역 CSV 업로드", type=["csv"])
        if manual_upload is not None and st.button("업로드 적용"):
            try:
                uploaded_df = pd.read_csv(manual_upload)
            except Exception as exc:
                st.error(f"CSV를 읽을 수 없습니다: {exc}")
            else:
                st.session_state["manual_tx_df"] = normalize_manual_tx_df(uploaded_df)
                st.session_state["manual_tx_last_digest"] = ""
                st.success("매수 내역을 불러왔습니다.")

        if st.button("매수 내역 초기화"):
            st.session_state["manual_tx_df"] = normalize_manual_tx_df(None)
            persist_error = persist_manual_tx_df(st.session_state["manual_tx_df"])
            if persist_error:
                st.warning(persist_error)
            st.rerun()

        manual_display_cols = ["date", "ticker", "name", "amount_base", "price_usd", "shares"]
        manual_display_df = st.session_state["manual_tx_df"].reindex(
            columns=manual_display_cols
        )
        manual_tx_df = st.data_editor(
            manual_display_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config=manual_column_config,
            key="manual_tx_editor",
        )
        st.session_state["manual_tx_df"] = normalize_manual_tx_df(manual_tx_df)

        st.divider()
        st.caption("백업/복원")
        backup_files = list_manual_tx_backups()
        if backup_files:
            backup_labels = [format_backup_label(path) for path in backup_files]
            selected_label = st.selectbox(
                "백업 선택",
                options=backup_labels,
                index=0,
                key="backup_selector",
            )
            selected_path = backup_files[backup_labels.index(selected_label)]
            if st.button("선택한 백업 복원"):
                restored_df, restore_error = load_manual_tx_file(selected_path)
                if restore_error:
                    st.warning(restore_error)
                else:
                    st.session_state["manual_tx_df"] = restored_df
                    st.session_state["manual_tx_last_digest"] = ""
                    st.success("선택한 백업으로 복원했습니다.")
                    st.rerun()
        else:
            st.caption("사용 가능한 백업이 없습니다.")

    holdings_tickers = extract_holdings_tickers(holdings_bytes)
    persist_error = persist_manual_tx_df(st.session_state["manual_tx_df"])
    if persist_error:
        st.warning(persist_error)
    manual_tickers = extract_manual_tickers(st.session_state.get("manual_tx_df"))
    tickers = sorted({a.ticker for a in assets}.union(holdings_tickers).union(manual_tickers))
    unknown_holdings = set(holdings_tickers) - {a.ticker for a in assets}
    if unknown_holdings:
        st.warning(
            "보유 CSV에 계획에 없는 티커가 있습니다: "
            + ", ".join(sorted(unknown_holdings))
            + ". 목표 비중은 비어 있습니다."
        )
    unknown_manual = set(manual_tickers) - {a.ticker for a in assets}
    if unknown_manual:
        st.warning(
            "직접 입력 내역에 계획에 없는 티커가 있습니다: "
            + ", ".join(sorted(unknown_manual))
            + ". 목표 비중은 비어 있습니다."
        )
    if not tickers:
        st.error("유효한 티커가 없습니다.")
        st.stop()

    manual_date_min = None
    manual_date_max = None
    if isinstance(manual_df, pd.DataFrame) and "date" in manual_df.columns:
        manual_dates = pd.to_datetime(manual_df["date"], errors="coerce").dropna()
        if not manual_dates.empty:
            manual_date_min = manual_dates.min().date()
            manual_date_max = manual_dates.max().date()
    history_start = start_date
    if manual_date_min and manual_date_min < history_start:
        history_start = manual_date_min
    price_start = history_start if history_start <= today else today - dt.timedelta(days=30)
    with st.spinner("가격 데이터를 불러오는 중..."):
        prices_usd = download_prices(tickers, price_start, today)
        fx_rates = None

    refresh_nonce = st.session_state.get("refresh_nonce", 0)
    if (
        use_prepost
        and refresh_nonce > 0
        and st.session_state.get("last_live_price_nonce") != refresh_nonce
    ):
        live_prices = fetch_latest_prices(tickers)
        if not live_prices.empty:
            st.session_state["last_live_prices"] = live_prices
        st.session_state["last_live_price_nonce"] = refresh_nonce

    live_prices = st.session_state.get("last_live_prices") if use_prepost else None
    if use_prepost and isinstance(live_prices, pd.Series) and not live_prices.empty:
        live_date = pd.Timestamp(today)
        prices_usd = prices_usd.copy()
        if live_date not in prices_usd.index:
            prices_usd.loc[live_date] = np.nan
        for ticker, value in live_prices.items():
            if ticker in prices_usd.columns and not pd.isna(value):
                prices_usd.at[live_date, ticker] = value
        prices_usd = prices_usd.sort_index()

    if not prices_usd.empty:
        st.session_state["last_prices_usd"] = prices_usd
    elif "last_prices_usd" in st.session_state:
        prices_usd = st.session_state["last_prices_usd"]
        st.warning("가격 데이터 갱신에 실패해 이전 데이터를 사용합니다.")

    if prices_usd.empty:
        st.error("가격 데이터가 없습니다. 티커나 네트워크를 확인하세요.")
        st.caption(
            f"요청 티커: {', '.join(tickers)} | 조회 구간: {price_start} ~ {today}"
        )
        if any(" " in ticker for ticker in tickers):
            st.warning("티커에 공백이 포함되어 있습니다. 실제 티커인지 확인하세요.")
        if price_start == today:
            st.info(
                "오늘 시작으로 조회하면 장이 끝나기 전에는 데이터가 비어 있을 수 있습니다. "
                "시작일을 어제로 바꾸거나 잠시 후 다시 시도하세요."
            )
        st.stop()

    fx_lock_rate = np.nan
    if base_currency == "KRW":
        if fx_rates is None or fx_rates.empty:
            if "last_fx_rates" in st.session_state:
                fx_rates = st.session_state["last_fx_rates"]
                st.warning("환율 데이터 갱신에 실패해 이전 데이터를 사용합니다.")
            else:
                st.error("환율 데이터가 없습니다. 네트워크나 환율 설정을 확인하세요.")
                st.stop()
        fx_lock_rate = resolve_fx_lock_rate(fx_rates, fx_mode, custom_fx_rate, start_date)
        if fx_mode != "live" and pd.isna(fx_lock_rate):
            st.error("고정 환율을 계산할 수 없습니다. 환율 데이터 또는 사용자 입력을 확인하세요.")
            st.stop()
    if fx_rates is not None and not fx_rates.empty:
        st.session_state["last_fx_rates"] = fx_rates

    if base_currency == "KRW":
        fx_rate_for_value = (
            fx_rates.ffill().iloc[-1] if fx_mode == "live" else fx_lock_rate
        )
    else:
        fx_rate_for_value = np.nan

    if use_manual_transactions or has_manual_rows:
        filled_manual = autofill_manual_tx_df(
            manual_df, fx_rates, fx_mode, fx_lock_rate, base_currency
        )
        if not filled_manual.equals(normalize_manual_tx_df(manual_df)):
            st.session_state["manual_tx_df"] = filled_manual
            manual_df = filled_manual
            st.rerun()

    planned_amounts = None
    if manual_date_min and manual_date_max:
        planned_trading_days = trading_days
        if apply_us_holidays and trading_days:
            trading_min = min(trading_days)
            trading_max = max(trading_days)
            if manual_date_min < trading_min or manual_date_max > trading_max:
                planned_trading_days = get_us_trading_days(manual_date_min, manual_date_max)
        planned_amounts = build_planned_amount_map(
            assets,
            manual_date_min,
            manual_date_max,
            monthly_contribution_effective,
            plan_mode,
            planned_trading_days,
        )

    manual_transactions, manual_warnings = parse_manual_transactions(
        st.session_state.get("manual_tx_df", pd.DataFrame()),
        base_currency,
        fx_rates,
        fx_mode,
        fx_lock_rate,
        planned_amounts,
        asset_name_map,
    )

    transactions = manual_transactions
    tx_warnings = manual_warnings

    if tx_warnings:
        for message in tx_warnings[:5]:
            st.warning(message)
        if len(tx_warnings) > 5:
            st.warning("추가 경고가 생략되었습니다.")

    if transactions.empty:
        st.warning("직접 입력 매수 내역이 없습니다. 매수 내역을 입력하세요.")

    positions_selected = positions_from_transactions(transactions)
    positions_actual, holdings_warnings = parse_actual_holdings(
        io.BytesIO(holdings_bytes) if holdings_bytes else None,
        base_currency,
        fx_rate_for_value,
    )
    if holdings_warnings:
        for message in holdings_warnings:
            st.warning(message)
    if include_actual and holdings_bytes:
        st.warning("직접 입력 내역과 보유 CSV를 함께 사용하면 중복 집계될 수 있습니다.")
    positions_combined = (
        combine_positions([positions_selected, positions_actual])
        if include_actual
        else positions_selected
    )

    summary = build_summary_from_positions(
        assets, positions_combined, prices_usd, fx_rate_for_value, base_currency
    )
    timeseries = build_daily_timeseries(
        transactions,
        prices_usd,
        fx_rates,
        base_currency,
        fx_mode,
        fx_lock_rate,
        history_start,
        today,
    )

    selected_invested = (
        positions_selected["cost_base"].sum() if not positions_selected.empty else 0.0
    )
    actual_invested = (
        positions_actual["cost_base"].sum() if (include_actual and not positions_actual.empty) else 0.0
    )
    if summary.empty:
        total_invested = 0.0
        total_value = 0.0
        total_profit = 0.0
        profit_pct = np.nan
    else:
        total_invested = summary["cost_base"].sum()
        total_value = summary["market_value_base"].sum()
        total_profit = total_value - total_invested
        profit_pct = total_profit / total_invested if total_invested else np.nan

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 투자금", format_currency(total_invested, base_currency))
    col2.metric("평가금액", format_currency(total_value, base_currency))
    col3.metric("손익", format_currency(total_profit, base_currency))
    col4.metric("수익률", f"{profit_pct:.2%}" if not pd.isna(profit_pct) else "-")

    if include_actual and not positions_actual.empty:
        extra1, extra2 = st.columns(2)
        extra1.metric("직접 입력 매입금", format_currency(selected_invested, base_currency))
        extra2.metric("실제 보유 매입금", format_currency(actual_invested, base_currency))

    st.subheader("리마인더")
    next_deposit = next_deposit_date(today, deposit_day)
    days_to_deposit = (next_deposit - today).days
    reminder_start = max(today, start_date)
    reminder_end = reminder_start + dt.timedelta(days=int(reminder_days))
    upcoming_reminders = build_upcoming_schedule(
        assets,
        reminder_start,
        reminder_end,
        monthly_contribution_effective,
        plan_mode,
        trading_days,
    )
    next_buy_rows = None
    next_buy_total = None
    if not upcoming_reminders.empty:
        next_buy_date = upcoming_reminders["date"].min()
        next_buy_rows = upcoming_reminders[upcoming_reminders["date"] == next_buy_date].copy()
        next_buy_total = next_buy_rows["planned_amount_base"].sum()

    rem_col1, rem_col2 = st.columns(2)
    rem_col1.metric(
        "다음 납입일",
        f"{next_deposit.isoformat()} (D-{days_to_deposit})" if days_to_deposit >= 0 else "-",
    )
    if next_buy_rows is not None:
        rem_col2.metric(
            "다음 매수일",
            f"{next_buy_rows['date'].iloc[0].date()} (총 {format_currency(next_buy_total, base_currency)})",
        )
    else:
        rem_col2.metric("다음 매수일", "-")

    if not upcoming_reminders.empty:
        if next_buy_rows is not None:
            st.write("다음 매수 상세")
            next_display = next_buy_rows.copy()
            next_display["date"] = next_display["date"].dt.date
            next_display["planned_amount_base"] = next_display["planned_amount_base"].map(
                lambda x: format_currency(x, base_currency)
            )
            st.dataframe(
                next_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date": "날짜",
                    "ticker": "티커",
                    "name": "이름",
                    "planned_amount_base": "예정 매수금",
                },
            )
        reminder_display = upcoming_reminders.copy()
        reminder_display["date"] = reminder_display["date"].dt.date
        reminder_display["planned_amount_base"] = reminder_display["planned_amount_base"].map(
            lambda x: format_currency(x, base_currency)
        )
        st.dataframe(
            reminder_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "date": "날짜",
                "ticker": "티커",
                "name": "이름",
                "planned_amount_base": "예정 매수금",
            },
        )

    st.subheader("보유 현황")
    holdings_tab, targets_tab = st.tabs(["보유 현황", "목표"])
    if not summary.empty:
        render_download_buttons(
            "보유 현황",
            summary,
            f"holdings_{today.isoformat()}",
            "holdings",
        )

    with holdings_tab:
        display = summary.copy()
        display["target_weight"] = display["target_weight"].map(
            lambda x: f"{x:.2%}" if not pd.isna(x) else "-"
        )
        display["actual_weight"] = display["actual_weight"].map(
            lambda x: f"{x:.2%}" if not pd.isna(x) else "-"
        )
        display["shares"] = display["shares"].map(lambda x: f"{x:,.4f}")
        display["avg_cost_base"] = display["avg_cost_base"].map(
            lambda x: format_unit_price(x, base_currency)
        )
        display["current_price_usd"] = display["current_price_usd"].map(
            lambda x: f"{x:,.2f} USD" if not pd.isna(x) else "-"
        )
        display["cost_base"] = display["cost_base"].map(
            lambda x: format_currency(x, base_currency)
        )
        display["market_value_base"] = display["market_value_base"].map(
            lambda x: format_currency(x, base_currency)
        )
        display["profit_base"] = display["profit_base"].map(
            lambda x: format_currency(x, base_currency)
        )
        display["profit_pct"] = display["profit_pct"].map(
            lambda x: f"{x:.2%}" if not pd.isna(x) else "-"
        )

        display_columns = [
            "ticker",
            "name",
            "target_weight",
            "actual_weight",
            "shares",
            "avg_cost_base",
            "current_price_usd",
            "cost_base",
            "market_value_base",
            "profit_base",
            "profit_pct",
        ]
        st.dataframe(
            display[display_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": "티커",
                "name": "이름",
                "target_weight": "목표 비중",
                "actual_weight": "실제 비중",
                "shares": "보유 수량",
                "avg_cost_base": f"평균단가({base_currency})",
                "current_price_usd": "현재가",
                "cost_base": "매입금",
                "market_value_base": "평가금",
                "profit_base": "손익",
                "profit_pct": "수익률",
            },
        )

    with targets_tab:
        targets = summary.copy()
        if not targets.empty:
            render_download_buttons(
                "목표 현황",
                targets,
                f"targets_{today.isoformat()}",
                "targets",
            )
        targets["target_weight"] = targets["target_weight"].map(
            lambda x: f"{x:.2%}" if not pd.isna(x) else "-"
        )
        targets["actual_weight"] = targets["actual_weight"].map(
            lambda x: f"{x:.2%}" if not pd.isna(x) else "-"
        )
        targets["target_amount_base"] = targets["target_amount_base"].map(
            lambda x: format_currency(x, base_currency) if x > 0 else "-"
        )
        targets["progress_amount"] = targets["progress_amount"].map(
            lambda x: f"{x:.2%}" if not pd.isna(x) else "-"
        )
        targets["remaining_amount"] = targets["remaining_amount"].map(
            lambda x: format_currency(x, base_currency) if not pd.isna(x) else "-"
        )
        targets["target_shares"] = targets["target_shares"].map(
            lambda x: f"{x:,.4f}" if x > 0 else "-"
        )
        targets["progress_shares"] = targets["progress_shares"].map(
            lambda x: f"{x:.2%}" if not pd.isna(x) else "-"
        )
        targets["remaining_shares"] = targets["remaining_shares"].map(
            lambda x: f"{x:,.4f}" if not pd.isna(x) else "-"
        )

        st.dataframe(
            targets[
                [
                    "ticker",
                    "name",
                    "target_weight",
                    "actual_weight",
                    "target_amount_base",
                    "progress_amount",
                    "remaining_amount",
                    "target_shares",
                    "progress_shares",
                    "remaining_shares",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": "티커",
                "name": "이름",
                "target_weight": "목표 비중",
                "actual_weight": "실제 비중",
                "target_amount_base": "목표 금액",
                "progress_amount": "달성률(금액)",
                "remaining_amount": "남은 금액",
                "target_shares": "목표 수량",
                "progress_shares": "달성률(수량)",
                "remaining_shares": "남은 수량",
            },
        )

    st.subheader("리밸런싱 가이드")
    rebalance = build_rebalance_plan(
        summary, assets, monthly_contribution_effective, today, trading_days
    )
    if rebalance.empty:
        st.write("현재 입력값으로 리밸런싱 가이드를 만들 수 없습니다.")
    else:
        render_download_buttons(
            "리밸런싱 가이드",
            rebalance,
            f"rebalance_{today.isoformat()}",
            "rebalance",
        )
        rebalance_display = rebalance.copy()
        rebalance_display["target_weight"] = rebalance_display["target_weight"].map(
            lambda x: f"{x:.2%}"
        )
        rebalance_display["current_value"] = rebalance_display["current_value"].map(
            lambda x: format_currency(x, base_currency)
        )
        rebalance_display["target_value"] = rebalance_display["target_value"].map(
            lambda x: format_currency(x, base_currency)
        )
        rebalance_display["gap_to_target"] = rebalance_display["gap_to_target"].map(
            lambda x: format_currency(x, base_currency)
        )
        rebalance_display["recommended_monthly_buy"] = rebalance_display[
            "recommended_monthly_buy"
        ].map(lambda x: format_currency(x, base_currency))
        rebalance_display["per_buy_amount"] = rebalance_display["per_buy_amount"].map(
            lambda x: format_currency(x, base_currency) if not pd.isna(x) else "-"
        )
        rebalance_display["next_buy_date"] = rebalance_display["next_buy_date"].map(
            lambda x: x.isoformat() if isinstance(x, dt.date) else "-"
        )
        st.dataframe(
            rebalance_display[
                [
                    "ticker",
                    "target_weight",
                    "current_value",
                    "target_value",
                    "gap_to_target",
                    "recommended_monthly_buy",
                    "per_buy_amount",
                    "next_buy_date",
                    "next_month_buys",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": "티커",
                "target_weight": "목표 비중",
                "current_value": "현재 가치",
                "target_value": "목표 가치",
                "gap_to_target": "부족/초과",
                "recommended_monthly_buy": "월 추천 매수",
                "per_buy_amount": "회차별 매수",
                "next_buy_date": "다음 매수일",
                "next_month_buys": "다음 달 매수 횟수",
            },
        )

    st.subheader("포트폴리오 추이")
    if include_actual and not positions_actual.empty:
        st.caption("차트는 업로드한 보유분(매수일 없음)을 제외합니다.")
    if not timeseries.empty:
        chart_df = timeseries.copy()
        chart_df.columns = [
            f"포트폴리오 가치 ({base_currency})",
            f"투자금 ({base_currency})",
            f"손익 ({base_currency})",
        ]
        st.line_chart(chart_df, height=300)

    st.subheader("납입/매수 리포트")
    st.caption("직접 입력 매수 기준입니다.")
    daily_tab, weekly_tab, monthly_tab = st.tabs(["일간", "주간", "월간"])

    for tab, period in [
        (daily_tab, "daily"),
        (weekly_tab, "weekly"),
        (monthly_tab, "monthly"),
    ]:
        with tab:
            totals, by_asset = build_period_report(transactions, period)
            if totals.empty:
                st.write("데이터가 없습니다.")
                continue
            period_label = {
                "daily": "일간",
                "weekly": "주간",
                "monthly": "월간",
            }[period]
            render_download_buttons(
                f"{period_label} 리포트(총괄)",
                totals,
                f"report_{period}_totals_{today.isoformat()}",
                f"report_{period}_totals",
            )
            render_download_buttons(
                f"{period_label} 리포트(자산별)",
                by_asset,
                f"report_{period}_by_asset_{today.isoformat()}",
                f"report_{period}_by_asset",
            )
            totals_display = totals.copy()
            totals_display["invested_base"] = totals_display["invested_base"].map(
                lambda x: format_currency(x, base_currency)
            )
            totals_display["shares"] = totals_display["shares"].map(lambda x: f"{x:,.6f}")
            st.dataframe(
                totals_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "period": "기간",
                    "buys": "매수 횟수",
                    "invested_base": "투자금",
                    "shares": "수량",
                },
            )

            by_asset_display = by_asset.copy()
            by_asset_display["invested_base"] = by_asset_display["invested_base"].map(
                lambda x: format_currency(x, base_currency)
            )
            by_asset_display["shares"] = by_asset_display["shares"].map(lambda x: f"{x:,.6f}")
            st.dataframe(
                by_asset_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "period": "기간",
                    "ticker": "티커",
                    "buys": "매수 횟수",
                    "invested_base": "투자금",
                    "shares": "수량",
                },
            )

    st.subheader("최근 매수 내역(직접 입력)")
    if transactions.empty:
        st.write("아직 매수 내역이 없습니다.")
    else:
        recent_tx = transactions.tail(20)
        recent_display = recent_tx[["date", "ticker", "amount_base", "price_usd", "shares"]].copy()
        recent_display["date"] = recent_display["date"].dt.date
        recent_display["amount_base"] = recent_display["amount_base"].map(
            lambda x: format_currency(x, base_currency)
        )
        recent_display["price_usd"] = recent_display["price_usd"].map(
            lambda x: f"{x:,.2f} USD" if not pd.isna(x) else "-"
        )
        recent_display["shares"] = recent_display["shares"].map(
            lambda x: f"{x:,.6f}" if not pd.isna(x) else "-"
        )
        st.dataframe(
            recent_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "date": "날짜",
                "ticker": "티커",
                "amount_base": "매수금",
                "price_usd": "가격(USD)",
                "shares": "수량",
            },
        )

    st.subheader("예정 매수 일정")
    schedule_start = max(today, start_date)
    future_end = schedule_start + dt.timedelta(days=30)
    upcoming = build_upcoming_schedule(
        assets,
        schedule_start,
        future_end,
        monthly_contribution_effective,
        plan_mode,
        trading_days,
    )
    if upcoming.empty:
        st.write("다음 30일 내 예정 매수가 없습니다.")
    else:
        upcoming_display = upcoming.copy()
        upcoming_display["date"] = upcoming_display["date"].dt.date
        upcoming_display["planned_amount_base"] = upcoming_display["planned_amount_base"].map(
            lambda x: format_currency(x, base_currency)
        )
        st.dataframe(
            upcoming_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "date": "날짜",
                "ticker": "티커",
                "name": "이름",
                "planned_amount_base": "예정 매수금",
            },
        )


if __name__ == "__main__":
    main()
