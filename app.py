"""
AI Data Analyst
Upload, clean, explore and chat with your data using Streamlit + Gemini.
"""

import io
import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from google.genai import types

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

DEFAULT_MODEL = "gemini-flash-latest"
MAX_CONTEXT_CHARS = 20000
MAX_SCATTER_POINTS = 20000
FREQUENCIES = {"Day": "D", "Week": "W", "Month": "MS", "Quarter": "QS", "Year": "YS"}

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        padding: 12px 16px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Column helpers
# ----------------------------------------------------------------------------
def numeric_columns(df):
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]


def datetime_columns(df):
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def text_columns(df):
    return [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]


def categorical_columns(df):
    skip = set(numeric_columns(df)) | set(datetime_columns(df))
    return [c for c in df.columns if c not in skip]


def unique_names(names):
    seen = {}
    result = []
    for name in names:
        if name in seen:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            result.append(name)
    return result


def arrow_safe(df):
    """Make mixed-type object columns displayable in st.dataframe."""
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        if out[col].dropna().map(type).nunique() > 1:
            out[col] = out[col].astype(str).where(out[col].notna(), np.nan)
    return out


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def init_state():
    defaults = {
        "raw": None,
        "df": None,
        "source": None,
        "log": [],
        "chat": [],
        "version": 0,
        "flash": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_df(new_df):
    st.session_state.df = new_df
    st.session_state.version += 1


def set_dataset(df, source):
    df = df.copy()
    df.columns = unique_names([str(c).strip() or "column" for c in df.columns])
    st.session_state.raw = df
    st.session_state.source = source
    st.session_state.log = []
    st.session_state.chat = []
    set_df(df.copy())


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def read_csv_bytes(data):
    last_error = None
    for encoding in ("utf-8-sig", "latin-1"):
        for kwargs in ({"sep": None, "engine": "python"}, {}):
            try:
                return pd.read_csv(io.BytesIO(data), encoding=encoding, **kwargs)
            except Exception as exc:  # try the next combination
                last_error = exc
    raise last_error


def make_sample():
    rng = np.random.default_rng(42)
    n = 500
    trend = 1 + np.arange(n) / n
    units = rng.integers(1, 20, n)
    price = np.clip(rng.normal(50, 15, n), 5, None).round(2)
    discount = rng.choice([0, 0.05, 0.1, 0.2], n)
    df = pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "region": rng.choice(["North", "South", "East", "West"], n),
            "category": rng.choice(["Electronics", "Clothing", "Home", "Sports"], n),
            "units": units,
            "unit_price": price,
            "discount": discount,
            "revenue": (units * price * (1 - discount) * trend).round(2),
            "customer_rating": rng.integers(1, 6, n).astype(float),
        }
    )
    df.loc[rng.choice(n, 25, replace=False), "customer_rating"] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "region"] = np.nan
    df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)
    return df


def sidebar():
    st.sidebar.title("📊 AI Data Analyst")
    uploaded = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded is not None:
        try:
            data = uploaded.getvalue()
            if uploaded.name.lower().endswith(".xlsx"):
                excel = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
                sheet = st.sidebar.selectbox("Excel sheet", excel.sheet_names)
                key = f"{uploaded.name}|{len(data)}|{sheet}"
                reader = lambda: pd.read_excel(excel, sheet_name=sheet)
            else:
                key = f"{uploaded.name}|{len(data)}"
                reader = lambda: read_csv_bytes(data)

            if st.session_state.source != key:
                loaded = reader()
                if loaded.empty:
                    st.sidebar.error("This file has no data rows.")
                else:
                    set_dataset(loaded, key)
        except Exception as exc:
            st.sidebar.error(f"Could not read the file: {exc}")
    elif st.sidebar.button("Load sample data"):
        set_dataset(make_sample(), "sample")

    st.sidebar.divider()
    model = st.sidebar.text_input(
        "Gemini model",
        value=DEFAULT_MODEL,
        help="Change this if Google retires or renames a model.",
    )

    if st.session_state.df is not None:
        if st.sidebar.button("↩️ Reset to original data"):
            set_df(st.session_state.raw.copy())
            st.session_state.log = []
            st.rerun()

    return model.strip() or DEFAULT_MODEL


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def apply_cleaning(df, opts):
    out = df.copy()
    notes = []

    if opts["strip"]:
        for col in text_columns(out):
            out[col] = out[col].map(lambda v: (v.strip() or np.nan) if isinstance(v, str) else v)
        notes.append("Trimmed whitespace in text columns and turned blank cells into missing values")

    if opts["dups"]:
        before = len(out)
        out = out.drop_duplicates()
        notes.append(f"Removed {before - len(out):,} duplicate rows")

    if opts["drop_cols"]:
        missing_pct = out.isna().mean() * 100
        to_drop = missing_pct[missing_pct > opts["max_missing"]].index.tolist()
        out = out.drop(columns=to_drop)
        detail = f": {', '.join(map(str, to_drop))}" if to_drop else ""
        notes.append(f"Dropped {len(to_drop)} column(s) with more than {opts['max_missing']}% missing{detail}")

    nums = numeric_columns(out)
    strategy = opts["num_strategy"]
    if strategy != "Leave as is" and nums:
        if strategy == "Drop rows":
            before = len(out)
            out = out.dropna(subset=nums)
            notes.append(f"Dropped {before - len(out):,} rows with missing numeric values")
        else:
            filled = int(out[nums].isna().sum().sum())
            for col in nums:
                if strategy == "Fill with mean":
                    value = out[col].mean()
                elif strategy == "Fill with median":
                    value = out[col].median()
                else:
                    value = 0
                out[col] = out[col].fillna(value)
            notes.append(f"{strategy} applied to numeric columns ({filled:,} cells)")

    cats = categorical_columns(out)
    strategy = opts["cat_strategy"]
    if strategy != "Leave as is" and cats:
        if strategy == "Drop rows":
            before = len(out)
            out = out.dropna(subset=cats)
            notes.append(f"Dropped {before - len(out):,} rows with missing text/category values")
        else:
            filled = int(out[cats].isna().sum().sum())
            for col in cats:
                if not out[col].isna().any():
                    continue
                if strategy == "Fill with most frequent value":
                    mode = out[col].mode(dropna=True)
                    if len(mode):
                        out[col] = out[col].fillna(mode.iloc[0])
                else:
                    out[col] = out[col].astype(object).fillna("Unknown")
            notes.append(f"{strategy} applied to text/category columns ({filled:,} cells)")

    if opts["outlier_cols"]:
        before = len(out)
        for col in opts["outlier_cols"]:
            if col in out.columns:
                q1, q3 = out[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                low, high = q1 - opts["iqr_k"] * iqr, q3 + opts["iqr_k"] * iqr
                out = out[out[col].between(low, high) | out[col].isna()]
        notes.append(f"Removed {before - len(out):,} outlier rows (IQR × {opts['iqr_k']}) in: {', '.join(map(str, opts['outlier_cols']))}")

    if opts["rename"]:
        names = [re.sub(r"\W+", "_", str(c).strip().lower()).strip("_") or "column" for c in out.columns]
        out.columns = unique_names(names)
        notes.append("Standardized column names")

    return out.reset_index(drop=True), notes


# ----------------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def csv_bytes(_df, version):
    return _df.to_csv(index=False).encode("utf-8")


def excel_bytes(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cleaned_data")
    return buffer.getvalue()


# ----------------------------------------------------------------------------
# Gemini
# ----------------------------------------------------------------------------
def get_api_key():
    try:
        key = st.secrets["GEMINI_API_KEY"]
        return str(key).strip() or None
    except Exception:
        return None


@st.cache_resource
def get_client(api_key):
    return genai.Client(api_key=api_key)


def build_context(df):
    parts = [
        f"Rows: {len(df)}, Columns: {df.shape[1]}",
        f"Duplicate rows: {int(df.duplicated().sum())}",
    ]
    cols = list(df.columns[:60])
    info = pd.DataFrame(
        {
            "dtype": df[cols].dtypes.astype(str),
            "missing": df[cols].isna().sum(),
            "unique": df[cols].nunique(),
        }
    )
    parts.append("Columns:\n" + info.to_string())

    nums = numeric_columns(df)[:40]
    if nums:
        parts.append("Numeric summary:\n" + df[nums].describe().T.round(3).to_string())
    if len(nums) >= 2:
        parts.append("Correlations (Pearson):\n" + df[nums[:15]].corr().round(2).to_string())
    parts.append("First 10 rows (CSV):\n" + df.head(10).to_csv(index=False))
    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


def build_system_prompt(df):
    return (
        "You are a senior data analyst assistant inside a Streamlit app. "
        "Answer using ONLY the dataset information below. Be concise and specific, "
        "mention column names and numbers, and use Markdown. If the answer cannot be "
        "determined from the summary or sample rows, say so and explain which analysis "
        "in the app (charts, correlations, trends) would answer it. Never invent values.\n\n"
        "DATASET CONTEXT\n" + build_context(df)
    )


def ask_gemini(api_key, model, df, question):
    st.session_state.chat.append({"role": "user", "content": question})
    history = st.session_state.chat[-11:]
    while history and history[0]["role"] != "user":
        history = history[1:]

    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in history
    ]
    try:
        client = get_client(api_key)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=build_system_prompt(df)),
        )
        text = response.text or "I could not generate an answer. Please rephrase your question."
        st.session_state.chat.append({"role": "assistant", "content": text})
        return None
    except Exception as exc:
        st.session_state.chat.pop()
        return str(exc)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
init_state()
model_name = sidebar()

st.title("📊 AI Data Analyst")
st.caption("Upload data, clean it, explore it visually and ask Gemini questions about it.")

df = st.session_state.df

if df is None:
    st.info("👈 Upload a CSV or Excel file in the sidebar, or click **Load sample data** to try the app.")
    st.markdown(
        """
        **What you can do here**
        - Preview data and profile every column
        - Find missing values and duplicates
        - Clean data with one click and download the result
        - Build interactive charts, correlations and trend views
        - Ask Gemini questions about your dataset
        """
    )
    st.stop()

if df.shape[0] == 0 or df.shape[1] == 0:
    st.warning("The dataset is empty. Use **Reset to original data** in the sidebar.")
    st.stop()

num_cols = numeric_columns(df)
cat_cols = categorical_columns(df)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", f"{df.shape[1]:,}")
m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
m4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
m5.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

(
    tab_overview,
    tab_quality,
    tab_clean,
    tab_viz,
    tab_stats,
    tab_trend,
    tab_ai,
    tab_download,
) = st.tabs(
    [
        "👀 Overview",
        "🔍 Data Quality",
        "🧹 Clean",
        "📈 Charts",
        "📐 Statistics & Correlation",
        "📅 Trends",
        "🤖 AI Assistant",
        "⬇️ Download",
    ]
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    st.subheader("Data preview")
    c1, c2 = st.columns([1, 2])
    n_rows = int(c1.number_input("Rows to show", min_value=1, max_value=1000, value=10, step=5, key="preview_rows"))
    view = c2.radio("Show", ["First rows", "Last rows", "Random sample"], horizontal=True, key="preview_mode")
    if view == "First rows":
        preview = df.head(n_rows)
    elif view == "Last rows":
        preview = df.tail(n_rows)
    else:
        preview = df.sample(min(n_rows, len(df)), random_state=1)
    st.dataframe(arrow_safe(preview))
    st.caption(f"{len(df):,} rows × {df.shape[1]} columns")

# ------------------------------------------------------------ Data quality
with tab_quality:
    st.subheader("Missing values")
    miss = pd.DataFrame({"Column": df.columns, "Missing": df.isna().sum().values})
    miss["Missing %"] = (miss["Missing"] / len(df) * 100).round(2)
    miss = miss[miss["Missing"] > 0].sort_values("Missing", ascending=False)
    if miss.empty:
        st.success("No missing values found 🎉")
    else:
        q1, q2 = st.columns([1, 2])
        q1.dataframe(miss, hide_index=True)
        fig = px.bar(miss, x="Column", y="Missing %", hover_data=["Missing"], title="Missing values by column")
        q2.plotly_chart(fig, key="missing_chart")

    st.subheader("Duplicate rows")
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        st.success("No duplicate rows found 🎉")
    else:
        st.warning(f"{dup_count:,} duplicate rows ({dup_count / len(df) * 100:.2f}% of all rows).")
        if st.checkbox("Show duplicated rows", key="show_dups"):
            st.dataframe(arrow_safe(df[df.duplicated(keep=False)].head(500)))

    st.subheader("Column profile")
    profile = pd.DataFrame(
        {
            "Column": df.columns,
            "Type": [str(t) for t in df.dtypes],
            "Non-null": df.notna().sum().values,
            "Missing": df.isna().sum().values,
            "Unique": df.nunique().values,
        }
    )
    st.dataframe(profile, hide_index=True)
    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant:
        st.info("Columns with a single value (little analytical use): " + ", ".join(map(str, constant)))

# ------------------------------------------------------------------- Clean
with tab_clean:
    st.subheader("Clean your data")
    if st.session_state.flash:
        st.success(st.session_state.flash)
        st.session_state.flash = None

    left, right = st.columns(2)
    with left:
        opt_strip = st.checkbox("Trim whitespace in text and turn blanks into missing values", value=True, key="opt_strip")
        opt_dups = st.checkbox("Remove duplicate rows", key="opt_dups")
        opt_drop_cols = st.checkbox("Drop columns with too many missing values", key="opt_drop_cols")
        opt_max_missing = st.slider(
            "Maximum missing % allowed per column", 0, 100, 50, key="opt_max_missing", disabled=not opt_drop_cols
        )
        opt_rename = st.checkbox("Standardize column names (lowercase_with_underscores)", key="opt_rename")
    with right:
        opt_num = st.selectbox(
            "Missing numeric values",
            ["Leave as is", "Fill with mean", "Fill with median", "Fill with 0", "Drop rows"],
            key="opt_num",
        )
        opt_cat = st.selectbox(
            "Missing text/category values",
            ["Leave as is", "Fill with most frequent value", "Fill with 'Unknown'", "Drop rows"],
            key="opt_cat",
        )
        opt_outliers = st.multiselect("Remove outliers (IQR rule) in these numeric columns", num_cols)
        opt_iqr = st.number_input("IQR multiplier", min_value=0.5, max_value=5.0, value=1.5, step=0.1, key="opt_iqr")

    if st.button("✅ Apply cleaning", type="primary", key="apply_clean"):
        options = {
            "strip": opt_strip,
            "dups": opt_dups,
            "drop_cols": opt_drop_cols,
            "max_missing": opt_max_missing,
            "rename": opt_rename,
            "num_strategy": opt_num,
            "cat_strategy": "Fill with 'Unknown'" if opt_cat == "Fill with 'Unknown'" else opt_cat,
            "outlier_cols": opt_outliers,
            "iqr_k": float(opt_iqr),
        }
        try:
            before_shape = df.shape
            cleaned, notes = apply_cleaning(df, options)
            if not notes:
                st.info("No cleaning options selected.")
            elif cleaned.shape[0] == 0 or cleaned.shape[1] == 0:
                st.error("These settings would remove all data, so nothing was changed. Try gentler options.")
            else:
                set_df(cleaned)
                summary = f"Shape: {before_shape[0]:,}×{before_shape[1]} → {cleaned.shape[0]:,}×{cleaned.shape[1]}"
                st.session_state.log.extend(notes + [summary])
                st.session_state.flash = "Cleaning applied. " + summary
                st.rerun()
        except Exception as exc:
            st.error(f"Cleaning failed: {exc}")

    st.divider()
    st.markdown("#### Change a column's type")
    t1, t2, t3 = st.columns([2, 2, 1])
    convert_col = t1.selectbox("Column to convert", list(df.columns))
    convert_to = t2.selectbox("Convert to", ["Number", "Date/time", "Text", "Category"], key="convert_to")
    t3.write("")
    t3.write("")
    if t3.button("Convert", key="convert_btn"):
        try:
            series = df[convert_col]
            if convert_to == "Number":
                converted = pd.to_numeric(series, errors="coerce")
            elif convert_to == "Date/time":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    converted = pd.to_datetime(series, errors="coerce")
            elif convert_to == "Text":
                converted = series.astype(str).where(series.notna(), np.nan)
            else:
                converted = series.astype("category")
            lost = int(converted.isna().sum() - series.isna().sum())
            new_df = df.copy()
            new_df[convert_col] = converted
            set_df(new_df)
            note = f"Converted '{convert_col}' to {convert_to}" + (f" ({lost:,} values could not be converted and are now missing)" if lost > 0 else "")
            st.session_state.log.append(note)
            st.session_state.flash = note
            st.rerun()
        except Exception as exc:
            st.error(f"Conversion failed: {exc}")

    if st.session_state.log:
        with st.expander("Cleaning history"):
            for line in st.session_state.log:
                st.markdown(f"- {line}")

# ------------------------------------------------------------------ Charts
with tab_viz:
    st.subheader("Interactive charts")
    chart_type = st.selectbox(
        "Chart type",
        ["Histogram", "Box plot", "Scatter plot", "Line chart", "Bar chart", "Pie chart"],
        key="chart_type",
    )
    color_options = ["None"] + list(df.columns)

    def pick(value):
        return None if value == "None" else value

    fig = None
    try:
        if chart_type == "Histogram":
            if not num_cols:
                st.info("This chart needs at least one numeric column.")
            else:
                a, b, c = st.columns(3)
                hx = a.selectbox("Histogram: column", num_cols)
                hc = b.selectbox("Histogram: color by", color_options)
                hb = c.slider("Histogram: bins", 5, 100, 30)
                fig = px.histogram(df, x=hx, color=pick(hc), nbins=hb, marginal="box")

        elif chart_type == "Box plot":
            if not num_cols:
                st.info("This chart needs at least one numeric column.")
            else:
                a, b = st.columns(2)
                by = a.selectbox("Box: value column", num_cols)
                bx = b.selectbox("Box: group by", ["None"] + list(df.columns))
                fig = px.box(df, y=by, x=pick(bx), color=pick(bx))

        elif chart_type == "Scatter plot":
            if len(num_cols) < 2:
                st.info("This chart needs at least two numeric columns.")
            else:
                a, b, c = st.columns(3)
                sx = a.selectbox("Scatter: X axis", num_cols)
                sy = b.selectbox("Scatter: Y axis", num_cols, index=1)
                sc = c.selectbox("Scatter: color by", color_options)
                plot_df = df
                if len(df) > MAX_SCATTER_POINTS:
                    plot_df = df.sample(MAX_SCATTER_POINTS, random_state=1)
                    st.caption(f"Showing a random sample of {MAX_SCATTER_POINTS:,} points for speed.")
                fig = px.scatter(plot_df, x=sx, y=sy, color=pick(sc), opacity=0.7)

        elif chart_type == "Line chart":
            if not num_cols:
                st.info("This chart needs at least one numeric column.")
            else:
                a, b, c = st.columns(3)
                lx = a.selectbox("Line: X axis", list(df.columns))
                ly = b.selectbox("Line: Y axis", num_cols)
                lc = c.selectbox("Line: color by", color_options)
                fig = px.line(df.sort_values(lx), x=lx, y=ly, color=pick(lc))

        elif chart_type == "Bar chart":
            a, b, c = st.columns(3)
            bar_x = a.selectbox("Bar: category column", list(df.columns))
            bar_agg = b.selectbox("Bar: measure", ["count", "sum", "mean", "median", "max", "min"])
            bar_n = c.slider("Bar: top categories", 5, 50, 15)
            if bar_agg == "count":
                counts = df[bar_x].astype(str).value_counts().head(bar_n)
                fig = px.bar(x=counts.index, y=counts.values, labels={"x": bar_x, "y": "count"})
            elif not num_cols:
                st.info("Pick 'count', or add a numeric column to aggregate.")
            else:
                bar_y = st.selectbox("Bar: numeric column", num_cols)
                grouped = df.groupby(bar_x)[bar_y].agg(bar_agg).sort_values(ascending=False).head(bar_n)
                fig = px.bar(
                    x=grouped.index.astype(str),
                    y=grouped.values,
                    labels={"x": bar_x, "y": f"{bar_agg} of {bar_y}"},
                )

        elif chart_type == "Pie chart":
            a, b = st.columns(2)
            pie_col = a.selectbox("Pie: category column", list(df.columns))
            pie_n = b.slider("Pie: top categories", 3, 20, 8)
            counts = df[pie_col].astype(str).value_counts().head(pie_n)
            fig = px.pie(names=counts.index, values=counts.values)

        if fig is not None:
            st.plotly_chart(fig, key="main_chart")
    except Exception as exc:
        st.error(f"Could not draw this chart: {exc}")

# ------------------------------------------------- Statistics & correlation
with tab_stats:
    st.subheader("Numeric summary")
    if not num_cols:
        st.info("No numeric columns found.")
    else:
        num_df = df[num_cols]
        summary = num_df.describe().T
        summary["missing"] = num_df.isna().sum()
        summary["skew"] = num_df.skew()
        summary["kurtosis"] = num_df.kurt()
        st.dataframe(summary.round(3))

    st.subheader("Text / category summary")
    if not cat_cols:
        st.info("No text or category columns found.")
    else:
        rows = []
        for col in cat_cols:
            counts = df[col].value_counts(dropna=True)
            rows.append(
                {
                    "Column": col,
                    "Unique": int(df[col].nunique(dropna=True)),
                    "Most frequent": str(counts.index[0]) if len(counts) else "",
                    "Frequency": int(counts.iloc[0]) if len(counts) else 0,
                    "Missing": int(df[col].isna().sum()),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.subheader("Correlation analysis")
    if len(num_cols) < 2:
        st.info("Correlation needs at least two numeric columns.")
    else:
        method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True, key="corr_method")
        try:
            corr = df[num_cols].corr(method=method)
            heat = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title=f"{method.title()} correlation matrix",
            )
            st.plotly_chart(heat, key="corr_heatmap")

            names = corr.columns.tolist()
            pairs = [
                (names[i], names[j], corr.iloc[i, j])
                for i in range(len(names))
                for j in range(i + 1, len(names))
                if pd.notna(corr.iloc[i, j])
            ]
            if pairs:
                pair_df = pd.DataFrame(pairs, columns=["Column A", "Column B", "Correlation"])
                pair_df = pair_df.reindex(pair_df["Correlation"].abs().sort_values(ascending=False).index)

                def strength(value):
                    value = abs(value)
                    if value >= 0.7:
                        return "Strong"
                    if value >= 0.4:
                        return "Moderate"
                    if value >= 0.2:
                        return "Weak"
                    return "Very weak"

                pair_df["Strength"] = pair_df["Correlation"].map(strength)
                pair_df["Correlation"] = pair_df["Correlation"].round(3)
                st.markdown("**Strongest relationships**")
                st.dataframe(pair_df.head(15), hide_index=True)
                st.caption("Correlation shows association, not causation.")
        except Exception as exc:
            st.error(f"Could not compute correlations: {exc}")

# ------------------------------------------------------------------ Trends
with tab_trend:
    st.subheader("Trend analysis")
    if not num_cols:
        st.info("Trend analysis needs at least one numeric column.")
    else:
        date_options = list(df.columns)
        dt_cols = datetime_columns(df)
        default_idx = date_options.index(dt_cols[0]) if dt_cols else 0

        a, b, c, d = st.columns(4)
        date_col = a.selectbox("Date/time column", date_options, index=default_idx)
        value_col = b.selectbox("Value column", num_cols)
        freq_label = c.selectbox("Group by", list(FREQUENCIES.keys()), index=2)
        agg = d.selectbox("Aggregate", ["sum", "mean", "median", "count", "min", "max"])
        window = st.slider("Moving average window (periods)", 1, 30, 3)

        if date_col == value_col:
            st.warning("Choose different columns for date and value.")
        else:
            try:
                tmp = df[[date_col, value_col]].copy()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                tmp = tmp.dropna(subset=[date_col])
                if tmp.empty:
                    st.warning("That column could not be read as dates. Pick a date column or convert it in the Clean tab.")
                else:
                    series = tmp.set_index(date_col)[value_col].resample(FREQUENCIES[freq_label]).agg(agg)
                    trend = pd.DataFrame({"period": series.index, "value": series.values})
                    trend["moving_avg"] = trend["value"].rolling(window, min_periods=1).mean()

                    label_main = f"{agg} of {value_col}"
                    label_ma = f"{window}-period moving average"
                    plot_trend = trend.rename(columns={"value": label_main, "moving_avg": label_ma})
                    fig = px.line(plot_trend, x="period", y=[label_main, label_ma], labels={"variable": "Series", "value": value_col, "period": date_col})
                    st.plotly_chart(fig, key="trend_chart")

                    valid = trend["value"].dropna()
                    if len(valid) >= 2:
                        slope = float(np.polyfit(np.arange(len(valid)), valid.values.astype(float), 1)[0])
                        first, last = float(valid.iloc[0]), float(valid.iloc[-1])
                        change = (last - first) / abs(first) * 100 if first != 0 else None
                        peak = trend.loc[trend["value"].idxmax()]
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Periods", f"{len(trend):,}")
                        k2.metric("Change first → last", f"{change:+.1f}%" if change is not None else "n/a")
                        k3.metric("Trend slope / period", f"{slope:,.3g}")
                        k4.metric("Peak period", str(pd.Timestamp(peak["period"]).date()))
                        direction = "upward 📈" if slope > 0 else "downward 📉" if slope < 0 else "flat ➖"
                        st.caption(f"The fitted linear trend is {direction}.")
            except Exception as exc:
                st.error(f"Could not build the trend: {exc}")

# ------------------------------------------------------------ AI assistant
with tab_ai:
    st.subheader("Gemini AI assistant")
    api_key = get_api_key()
    if not api_key:
        st.error(
            "Gemini API key not found. Add `GEMINI_API_KEY` in your app's Secrets "
            "(Streamlit Cloud → App settings → Secrets). See the README."
        )
    else:
        st.caption(
            "Gemini receives your column names, summary statistics and the first 10 rows — "
            "not the full dataset. Avoid uploading sensitive data."
        )
        history_box = st.container()

        prompts = {
            "📋 Summarize dataset": "Give me a clear summary of this dataset: what it seems to contain, its size, the key columns and the most notable patterns.",
            "🧹 Data quality issues": "Review the data quality: missing values, duplicates, suspicious types, constant columns and possible outliers. Recommend specific cleaning steps.",
            "💡 Suggest analyses": "Suggest the 5 most valuable analyses or charts for this dataset and explain what each would reveal.",
        }
        question = None
        b1, b2, b3, b4 = st.columns(4)
        for column, (label, text) in zip((b1, b2, b3), prompts.items()):
            if column.button(label, key=f"quick_{label}"):
                question = text
        if b4.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state.chat = []
            st.rerun()

        with st.form("ask_form", clear_on_submit=True):
            typed = st.text_input("Ask a question about your data", placeholder="e.g. Which region has the highest average revenue?")
            sent = st.form_submit_button("Send")
        if sent and typed.strip():
            question = typed.strip()

        error_message = None
        if question:
            with st.spinner("Gemini is analysing your data..."):
                error_message = ask_gemini(api_key, model_name, df, question)

        with history_box:
            if not st.session_state.chat:
                st.info("Ask a question below or use a quick action.")
            for message in st.session_state.chat:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        if error_message:
            st.error(f"Gemini request failed: {error_message}")
            st.caption("If the error mentions the model, change the model name in the sidebar.")

# ---------------------------------------------------------------- Download
with tab_download:
    st.subheader("Download cleaned data")
    st.write(f"Current dataset: **{len(df):,} rows × {df.shape[1]} columns**")
    st.download_button(
        "⬇️ Download cleaned data (CSV)",
        data=csv_bytes(df, st.session_state.version),
        file_name="cleaned_data.csv",
        mime="text/csv",
        key="download_csv",
    )
    if st.checkbox("Also prepare an Excel (.xlsx) file", key="prep_excel"):
        try:
            st.download_button(
                "⬇️ Download cleaned data (Excel)",
                data=excel_bytes(df),
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_xlsx",
            )
        except Exception as exc:
            st.error(f"Could not create the Excel file: {exc}")
