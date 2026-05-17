import streamlit as st
import pandas as pd
import joblib
import math

# Page config
st.set_page_config(
    page_title="Laptopia",
    page_icon="💻",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.laptopia-header {
    text-align: center;
    padding: 3rem 0 1.5rem;
}
.laptopia-header h1 {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #ffffff 30%, #a3e4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.laptopia-header p {
    color: #888;
    font-size: 1rem;
    margin-top: 0.4rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 1px solid #222;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    color: #555;
    background: transparent;
    border: none;
    padding: 0.6rem 1.4rem;
    border-radius: 8px 8px 0 0;
    letter-spacing: 0.5px;
}
.stTabs [aria-selected="true"] {
    color: #fff !important;
    background: #1a1a1a !important;
    border-bottom: 2px solid #a3e4ff !important;
}

.stTextInput input {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    color: #f0f0f0;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    padding: 0.75rem 1rem;
}
.stTextInput input:focus {
    border-color: #a3e4ff;
    box-shadow: 0 0 0 2px rgba(163,228,255,0.1);
}

.laptop-card {
    background: #141414;
    border: 1px solid #222;
    border-radius: 14px;
    padding: 1rem;
    height: 100%;
}
.laptop-card img {
    width: 100%;
    height: 150px;
    object-fit: contain;
    border-radius: 8px;
    background: #1e1e1e;
    margin-bottom: 0.75rem;
}
.card-no-img {
    width: 100%;
    height: 150px;
    background: #1e1e1e;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #444;
    font-size: 0.8rem;
    margin-bottom: 0.75rem;
}
.card-name {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    color: #f0f0f0;
    margin-bottom: 0.5rem;
    line-height: 1.3;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.card-badge {
    display: inline-block;
    background: #1e2d38;
    color: #a3e4ff;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 0.5rem;
}
.card-spec {
    font-size: 0.75rem;
    color: #888;
    margin: 2px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.card-spec span { color: #ccc; font-weight: 500; }
.card-price {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #a3e4ff;
    margin-top: 0.6rem;
}
.card-rating { font-size: 0.75rem; color: #f5c542; margin-top: 2px; }

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 1rem;
}
.rec-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #a3e4ff;
    padding: 1rem 0 0.5rem;
    border-top: 1px solid #222;
    margin-top: 1.5rem;
}
.page-info {
    text-align: center;
    color: #555;
    font-size: 0.8rem;
    margin-top: 1rem;
}

section[data-testid="stSidebar"] {
    background: #111;
    border-right: 1px solid #1e1e1e;
}
section[data-testid="stSidebar"] .stCheckbox label {
    color: #ccc;
    font-size: 0.85rem;
}
.filter-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #555;
    margin: 1.2rem 0 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Load data & model
@st.cache_data
def load_data():
    return pd.read_csv("laptop_cleaned.csv")

@st.cache_resource
def load_model():
    return joblib.load("cosine_sim.pkl")

df         = load_data()
cosine_sim = load_model()
df         = df.reset_index(drop=True)

# Header
st.markdown("""
<div class="laptopia-header">
    <h1>💻 Laptopia</h1>
    <p>Find your next laptop — search or discover by preference.</p>
</div>
""", unsafe_allow_html=True)

# Helper: build card HTML
def card_html(row):
    img_url = str(row.get("img", ""))
    name    = row.get("name", "Unknown")
    gpu     = str(row.get("graphics_card", "-"))
    ram     = row.get("ram_gb", "-")
    storage = row.get("storage_gb", "-")
    price   = row.get("price_idr", 0)
    rating  = row.get("rating", "-")
    cat     = row.get("laptop_category", "")
    proc    = str(row.get("processor", "-"))

    proc_disp = proc if len(proc) <= 38 else proc[:38] + "…"
    gpu_disp  = gpu  if len(gpu)  <= 38 else gpu[:38]  + "…"
    price_fmt = f"Rp {int(price):,}".replace(",", ".")

    stars = ""
    try:
        stars = "★" * int(float(rating))
    except Exception:
        pass

    uid = row.name  # unique row index for onerror id
    if img_url and img_url != "nan":
        img_block = (
            f'<img src="{img_url}" '
            f'onerror="this.style.display=\'none\';'
            f'document.getElementById(\'ni_{uid}\').style.display=\'flex\'">'
            f'<div class="card-no-img" id="ni_{uid}" style="display:none">Image not available</div>'
        )
    else:
        img_block = '<div class="card-no-img">Image not available</div>'

    return f"""
    <div class="laptop-card">
        {img_block}
        <div class="card-badge">{cat}</div>
        <div class="card-name">{name}</div>
        <div class="card-spec">Proc: <span>{proc_disp}</span></div>
        <div class="card-spec">GPU: <span>{gpu_disp}</span></div>
        <div class="card-spec">RAM: <span>{ram} GB</span> &nbsp;|&nbsp; Storage: <span>{storage} GB</span></div>
        <div class="card-price">{price_fmt}</div>
        <div class="card-rating">{stars} {rating}</div>
    </div>
    """

# Helper: paginated grid
def render_grid(dataframe, page_key, show_buttons=True):
    per_page    = 8
    total       = len(dataframe)
    total_pages = max(1, math.ceil(total / per_page))

    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    if st.session_state[page_key] > total_pages:
        st.session_state[page_key] = 1

    page  = st.session_state[page_key]
    start = (page - 1) * per_page
    chunk = dataframe.iloc[start:start + per_page]

    for row_start in range(0, len(chunk), 4):
        row_chunk = chunk.iloc[row_start:row_start + 4]
        cols = st.columns(4)
        for j, (_, row) in enumerate(row_chunk.iterrows()):
            with cols[j]:
                st.markdown(card_html(row), unsafe_allow_html=True)
                if show_buttons:
                    if st.button("See Recommendations →", key=f"rec_{page_key}_{row.name}", use_container_width=True):
                        st.session_state["active_rec"]      = int(row.name)
                        st.session_state["active_rec_name"] = str(row.get("name", ""))

    st.markdown(
        f'<div class="page-info">Page {page} of {total_pages} &nbsp;·&nbsp; {total} laptops found</div>',
        unsafe_allow_html=True,
    )

    col_prev, _, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("← Prev", disabled=(page <= 1), key=f"prev_{page_key}"):
            st.session_state[page_key] = page - 1
            st.rerun()
    with col_next:
        if st.button("Next →", disabled=(page >= total_pages), key=f"next_{page_key}"):
            st.session_state[page_key] = page + 1
            st.rerun()

# Helper: recommendation row
def show_recommendations(laptop_idx, laptop_name, top_n=8):
    scores  = list(enumerate(cosine_sim[laptop_idx]))
    scores  = sorted(scores, key=lambda x: x[1], reverse=True)
    scores  = [s for s in scores if s[0] != laptop_idx][:top_n]
    rec_idx = [s[0] for s in scores]

    st.markdown(
        f'<div class="rec-header">Similar laptops to "{laptop_name}"</div>',
        unsafe_allow_html=True,
    )

    for row_start in range(0, len(rec_idx), 4):
        chunk = rec_idx[row_start:row_start + 4]
        cols  = st.columns(4)
        for j, idx in enumerate(chunk):
            with cols[j]:
                st.markdown(card_html(df.iloc[idx]), unsafe_allow_html=True)

# Sidebar filters
with st.sidebar:
    st.markdown(
        '<div style="font-family:Syne;font-size:1.1rem;font-weight:700;color:#f0f0f0;padding:1rem 0 0.5rem">Filters</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<small style='color:#555'>Used in the Recommend tab.</small>",
        unsafe_allow_html=True,
    )

    # Laptop category
    st.markdown('<div class="filter-title">Laptop Category</div>', unsafe_allow_html=True)
    cat_order = ["Daily", "Business", "Gaming", "Ultrabook", "Workstation", "MacBook", "Chromebook", "2-in-1"]
    avail_cat = [c for c in cat_order if c in df["laptop_category"].values]
    sel_cat   = [c for c in avail_cat if st.checkbox(c, key=f"cat_{c}")]

    # Brand
    st.markdown('<div class="filter-title">Brand</div>', unsafe_allow_html=True)
    all_brands = sorted(df["brand"].dropna().unique().tolist())
    sel_brands = [b for b in all_brands if st.checkbox(b, key=f"brand_{b}")]

    # Processor brand
    st.markdown('<div class="filter-title">Processor Brand</div>', unsafe_allow_html=True)
    all_proc = sorted(df["processor_brand"].dropna().unique().tolist())
    sel_proc = [p for p in all_proc if st.checkbox(p, key=f"proc_{p}")]

    # Price range
    st.markdown('<div class="filter-title">Price Range</div>', unsafe_allow_html=True)
    price_order = ["Low", "Entry", "Premium", "Flagship"]
    avail_price = [p for p in price_order if p in df["price_category"].values]
    sel_price   = [p for p in avail_price if st.checkbox(p, key=f"price_{p}")]

    # GPU type
    st.markdown('<div class="filter-title">GPU Type</div>', unsafe_allow_html=True)
    all_gpu = sorted(df["gpu_type"].dropna().unique().tolist())
    sel_gpu = [g for g in all_gpu if st.checkbox(g, key=f"gpu_{g}")]

    # OS
    st.markdown('<div class="filter-title">Operating System</div>', unsafe_allow_html=True)
    os_clean_map = {
        "Windows 11 OS"  : "Windows 11",
        "Windows 11 Home": "Windows 11",
        "Windows 10 OS"  : "Windows 10",
        "Windows OS"     : "Windows",
        "Mac OS"         : "macOS",
        "Chrome OS"      : "Chrome OS",
        "HarmonyOS 5"    : "HarmonyOS",
        "DOS OS"         : "DOS",
    }
    df["os_display"] = df["os"].map(os_clean_map).fillna(df["os"])
    all_os  = sorted(df["os_display"].dropna().unique().tolist())
    sel_os  = [o for o in all_os if st.checkbox(o, key=f"os_{o}")]

    # Disk type
    st.markdown('<div class="filter-title">Disk Type</div>', unsafe_allow_html=True)
    all_disk = sorted(df["disk_type"].dropna().unique().tolist())
    sel_disk = [d for d in all_disk if st.checkbox(d, key=f"disk_{d}")]


tab_search, tab_filter = st.tabs(["🔍  Search", "🎛️  Recommend by Filter"])


# TAB 1 — SEARCH
with tab_search:
    query = st.text_input(
        "",
        placeholder="Search laptop — e.g. 'Dell G15', 'Asus ROG', 'MacBook Pro'...",
    )

    if query.strip():
        results = df[df["name"].str.contains(query.strip(), case=False, na=False)]

        if results.empty:
            st.markdown(
                f"<p style='color:#555;padding:2rem 0'>No laptop found for <b>'{query}'</b>.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="section-label">{len(results)} result(s) for "{query}"</div>',
                unsafe_allow_html=True,
            )
            render_grid(results, page_key="search_page")

    # Recommendation section appears after clicking "See Recommendations"
    if "active_rec" in st.session_state:
        show_recommendations(
            st.session_state["active_rec"],
            st.session_state.get("active_rec_name", "selected laptop"),
        )

# TAB 2 — RECOMMEND BY FILTER
with tab_filter:
    filtered = df.copy()

    if sel_cat    : filtered = filtered[filtered["laptop_category"].isin(sel_cat)]
    if sel_brands : filtered = filtered[filtered["brand"].isin(sel_brands)]
    if sel_proc   : filtered = filtered[filtered["processor_brand"].isin(sel_proc)]
    if sel_price  : filtered = filtered[filtered["price_category"].isin(sel_price)]
    if sel_gpu    : filtered = filtered[filtered["gpu_type"].isin(sel_gpu)]
    if sel_os     : filtered = filtered[filtered["os_display"].isin(sel_os)]
    if sel_disk   : filtered = filtered[filtered["disk_type"].isin(sel_disk)]

    any_filter = any([sel_cat, sel_brands, sel_proc, sel_price, sel_gpu, sel_os, sel_disk])

    if not any_filter:
        st.markdown(
            "<p style='color:#555;padding:2rem 0'>Select at least one filter from the sidebar to get recommendations.</p>",
            unsafe_allow_html=True,
        )
    elif filtered.empty:
        st.markdown(
            "<p style='color:#555;padding:2rem 0'>No laptop matches the selected filters. Try removing some.</p>",
            unsafe_allow_html=True,
        )
    else:
        filtered_sorted = filtered.sort_values("rating", ascending=False)
        st.markdown(
            f'<div class="section-label">{len(filtered_sorted)} laptop(s) match your filters</div>',
            unsafe_allow_html=True,
        )
        render_grid(filtered_sorted, page_key="filter_page", show_buttons=False)