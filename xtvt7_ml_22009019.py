import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
import altair as alt

# -----------------------------
# Page Config and Theming
# -----------------------------
st.set_page_config(
    page_title="Iris Explorer",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply a custom Altair theme to match Streamlit theme
alt.theme.enable("opaque")

# -----------------------------
# Data Loading
# -----------------------------
@st.cache_data
def load_iris_df() -> pd.DataFrame:
    iris = datasets.load_iris(as_frame=True)
    df = iris.frame.copy()
    # Rename for clarity
    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]
    return df

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.title("Filters")
source_choice = st.sidebar.radio(
    "Data source",
    ["Sample: Iris (sklearn)", "Upload CSV", "Public CSV URL"],
    help="Choose where to load the dataset from",
)

if source_choice == "Sample: Iris (sklearn)":
    df = load_iris_df()
else:
    if source_choice == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"]) 
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.sidebar.error(f"Failed to read CSV: {e}")
                st.stop()
        else:
            st.sidebar.info("Please upload a CSV to continue.")
            st.stop()
    else:  # Public CSV URL
        url = st.sidebar.text_input(
            "Enter public CSV URL",
            value="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        )
        if url:
            try:
                df = pd.read_csv(url)
            except Exception as e:
                st.sidebar.error(f"Failed to fetch URL: {e}")
                st.stop()
        else:
            st.sidebar.info("Provide a CSV URL to continue.")
            st.stop()

# Ensure we have a DataFrame
if not isinstance(df, pd.DataFrame) or df.empty:
    st.error("No data available to display.")
    st.stop()

# Try to standardize species column name for Iris-like datasets
for cand in ["species", "variety", "class", "target"]:
    if cand in df.columns:
        species_col = cand
        break
else:
    species_col = None

# Numeric and categorical columns detection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Sidebar: Species/category filter (if categorical exists)
if species_col:
    categories = sorted(df[species_col].dropna().unique())
    selected_categories = st.sidebar.multiselect(
        f"Filter by {species_col}", options=categories, default=categories
    )
    if selected_categories:
        df = df[df[species_col].isin(selected_categories)]
else:
    selected_categories = []

# Sidebar: Row sampling / range filter
max_rows = int(df.shape[0])
row_count = st.sidebar.slider(
    "Rows to display (sample if needed)",
    min_value=min(50, max_rows if max_rows > 0 else 50),
    max_value=max_rows if max_rows > 0 else 50,
    value=min(150, max_rows),
)
if df.shape[0] > row_count:
    df_viz = df.sample(row_count, random_state=42)
else:
    df_viz = df.copy()

# Sidebar: Choose numeric columns for plots
if len(numeric_cols) >= 2:
    x_axis = st.sidebar.selectbox("X-axis", numeric_cols, index=0)
    y_axis = st.sidebar.selectbox("Y-axis", numeric_cols, index=1)
else:
    st.sidebar.warning("Need at least two numeric columns for scatter plot.")
    x_axis = numeric_cols[0] if numeric_cols else None
    y_axis = numeric_cols[0] if numeric_cols else None

# -----------------------------
# Header and Data Summary
# -----------------------------
st.title("🌸 Iris & CSV Explorer")

colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Rows", f"{df.shape[0]:,}")
with colB:
    st.metric("Columns", f"{df.shape[1]:,}")
with colC:
    st.metric("Numeric cols", f"{len(numeric_cols)}")
with colD:
    st.metric("Categorical cols", f"{len(cat_cols)}")

with st.expander("Preview Data", expanded=True):
    st.dataframe(df.head(50), use_container_width=True)

# -----------------------------
# Visualizations
# -----------------------------
st.subheader("Visualizations")

# 1) Scatter Plot (Altair)
if x_axis and y_axis and x_axis in df_viz.columns and y_axis in df_viz.columns:
    color_field = species_col if species_col in df_viz.columns else None
    tooltip_fields = [x_axis, y_axis]
    if color_field:
        tooltip_fields.append(color_field)

    scatter = (
        alt.Chart(df_viz)
        .mark_circle(size=80, opacity=0.7)
        .encode(
            x=alt.X(x_axis, title=x_axis.replace("_", " ").title()),
            y=alt.Y(y_axis, title=y_axis.replace("_", " ").title()),
            color=alt.Color(color_field, legend=alt.Legend(title=color_field)) if color_field else alt.value("#4C78A8"),
            tooltip=tooltip_fields,
        )
        .interactive()
        .properties(height=420)
    )
    st.altair_chart(scatter, use_container_width=True)
else:
    st.info("Scatter plot requires at least two numeric columns.")

# 2) Histogram / Distribution
if numeric_cols:
    hist_col = st.selectbox("Choose column for histogram", numeric_cols, index=0)
    bins = st.slider("Bins", min_value=5, max_value=60, value=20)
    hist = (
        alt.Chart(df_viz)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X(f"{hist_col}:Q", bin=alt.Bin(maxbins=bins), title=hist_col.replace("_", " ").title()),
            y=alt.Y("count()", title="Count"),
            color=alt.Color(species_col) if species_col else alt.value("#F58518"),
            tooltip=[alt.Tooltip(hist_col, title=hist_col), alt.Tooltip("count()", title="Count")],
        )
        .properties(height=420)
    )
    st.altair_chart(hist, use_container_width=True)
else:
    st.info("No numeric columns found for histogram.")

# 3) Optional: Category count bar chart (if categorical column exists)
if species_col:
    st.subheader(f"{species_col.title()} Counts")
    count_df = df_viz[species_col].value_counts().reset_index()
    count_df.columns = [species_col, "count"]
    bar = (
        alt.Chart(count_df)
        .mark_bar()
        .encode(
            x=alt.X(species_col, sort='-y', title=species_col.title()),
            y=alt.Y("count", title="Count"),
            color=alt.Color(species_col, legend=None),
            tooltip=[species_col, "count"],
        )
        .properties(height=360)
    )
    st.altair_chart(bar, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.caption(
    "Built with Streamlit • Tip: use the sidebar to change data source, filter categories, select axes, and adjust bins."
)
