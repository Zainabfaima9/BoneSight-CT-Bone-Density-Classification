import os

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG — MUST BE BEFORE ANY STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="BoneSight-CT",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

RESULTS_CSV = "bonesight_results.csv"

# Pickhardt et al. 2019 thresholds (validated against DXA, >20,000 patients)
OSTEOPOROSIS_MAX = 110
OSTEOPENIA_MAX = 160

# Three real patients from the 20 successful results — one per category —
# used as sample/demo cases since live TotalSegmentator processing is too
# heavy to run inside the deployed app itself.
DEMO_CASES = [
    {
        "id": "Lung_Dx-A0098",
        "mean_hu": 71.65,
        "label": "Osteoporosis example",
        "image": "sample_images/Lung_Dx-A0098_overlay.png",
    },
    {
        "id": "Lung_Dx-A0080",
        "mean_hu": 127.11,
        "label": "Osteopenia example",
        "image": "sample_images/Lung_Dx-A0080_overlay.png",
    },
    {
        "id": "Lung_Dx-A0093",
        "mean_hu": 183.03,
        "label": "Normal example",
        "image": "sample_images/Lung_Dx-A0093_overlay.png",
    },
]


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f6f8fb;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ---------- FORCE READABLE TEXT ----------
       Streamlit's dark theme (if the user's browser/device defaults to
       it) makes plain text white-on-white against our light background.
       We force dark text everywhere so this can't happen. */

    .stApp, .stApp p, .stApp li, .stApp span, .stApp label,
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"],
    div[data-testid="stMetricDelta"], .stCaption, .stAlert p {
        color: #17324d !important;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #102a43 !important;
    }

    /* Container/card backgrounds — make sure they're light, not dark-theme black */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
    }

    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #102a43;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 0.76rem;
        color: #64748b;
        margin-top: 0.15rem;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 2.55rem;
        font-weight: 650;
        border: 1px solid #d8e2ea;
        background: white;
        color: #17324d;
    }

    div.stButton > button:hover {
        border-color: #a4785a;
        color: #a4785a;
    }

    h1 {
        color: #102a43 !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em;
    }

    h2, h3 { color: #17324d !important; }

    .small-muted {
        color: #64748b;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    .score-number {
        font-size: 3.2rem;
        font-weight: 800;
        color: #102a43;
        line-height: 1;
    }

    .score-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .result-box {
        padding: 1rem;
        border-radius: 12px;
        line-height: 1.6;
    }

    @media (max-width: 700px) {
        .main .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
            padding-top: 1rem;
        }
        .brand-title { font-size: 1rem; }
        .brand-subtitle { font-size: 0.7rem; }
        h1 { font-size: 2rem !important; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"


def navigate(page):
    st.session_state.page = page
    st.rerun()


# ============================================================
# DATA LOADING
# Cached so the CSV is only read once per session, not on every click.
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(RESULTS_CSV)
    return df

try:
    df = load_data()
    success_df = df[df["status"] == "success"].copy()
    DATA_READY = True
except FileNotFoundError:
    df = None
    success_df = None
    DATA_READY = False


# ============================================================
# CLASSIFICATION LOGIC
# ============================================================

DXA_BLURB = (
    "**What's a DXA scan?** The gold-standard test for bone density — a "
    "low-radiation scan (usually of the hip/spine) that a doctor uses to "
    "confirm osteoporosis or osteopenia and decide on treatment."
)


def classify_hu(hu_value):
    if hu_value <= OSTEOPOROSIS_MAX:
        return "Osteoporosis", "🔴", "#b3261e"
    elif hu_value <= OSTEOPENIA_MAX:
        return "Osteopenia", "🟠", "#a4785a"
    else:
        return "Normal", "🟢", "#22743a"


def interpretation(category):
    if category == "Osteoporosis":
        return {
            "definition": (
                "Osteoporosis means bone has lost density and strength "
                "faster than the body can rebuild it — bones become "
                "porous and fragile, with a much higher fracture risk."
            ),
            "action": "Refer for a formal DXA scan",
            "message": (
                "The HU value falls at or below the Osteoporosis "
                "threshold. This is a screening flag, not a "
                "diagnosis — a formal DXA scan is the next step."
            ),
        }
    elif category == "Osteopenia":
        return {
            "definition": (
                "Osteopenia means bone density is lower than normal, "
                "but not low enough to be called osteoporosis — an "
                "early warning stage, not a disease itself."
            ),
            "action": "Consider a DXA scan, especially with other risk factors",
            "message": (
                "The HU value falls in the Osteopenia range. Bone "
                "density may be lower than normal; a DXA scan can "
                "confirm this, particularly if other risk factors "
                "(age, fracture history, menopause) are present."
            ),
        }
    else:
        return {
            "definition": (
                "This range is consistent with normal, healthy bone "
                "density by this screening method."
            ),
            "action": "No additional action indicated by this signal",
            "message": (
                "The HU value falls above the Osteopenia threshold, "
                "consistent with normal bone density by this "
                "screening method."
            ),
        }


# ============================================================
# TOP NAVIGATION
# ============================================================

brand, home_btn, analyze_btn, demos_btn, results_btn, learn_btn = st.columns(
    [2.2, 1, 1, 1, 1, 1],
    vertical_alignment="center",
)

with brand:
    st.markdown('<div class="brand-title">🦴 BoneSight-CT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Opportunistic bone density screening from chest CT</div>',
        unsafe_allow_html=True,
    )

with home_btn:
    if st.button("Home", use_container_width=True):
        navigate("Home")

with analyze_btn:
    if st.button("Try It", use_container_width=True):
        navigate("Analyze")

with demos_btn:
    if st.button("Demos", use_container_width=True):
        navigate("Demos")

with results_btn:
    if st.button("Results", use_container_width=True):
        navigate("Results")

with learn_btn:
    if st.button("Learn", use_container_width=True):
        navigate("Learn")

st.divider()


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":

    st.markdown("### MEDICAL IMAGING TECHNOLOGY × AI")
    st.title("BoneSight-CT")

    st.write(
        "Millions of chest CT scans are performed every year for lung and "
        "heart reasons — and almost all of them already capture the L1 "
        "vertebra. BoneSight-CT reuses that existing scan to flag possible "
        "osteoporosis or osteopenia, at zero extra cost or radiation, in "
        "places where a dedicated DXA (bone density) scanner is unavailable."
    )

    st.write("")

    if DATA_READY:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Patients screened", len(success_df))
        with kpi2:
            st.metric("Flagged Osteoporosis", int((success_df["classification"] == "Osteoporosis").sum()))
        with kpi3:
            st.metric("Flagged Osteopenia", int((success_df["classification"] == "Osteopenia").sum()))
        with kpi4:
            st.metric("Normal", int((success_df["classification"] == "Normal").sum()))

        st.caption(
            "Results from a public chest CT dataset (TCIA Lung-PET-CT-Dx). "
            "See Results for the full breakdown and Learn for methodology."
        )
    else:
        st.warning(f"{RESULTS_CSV} not found — KPI cards will appear once it's added to the app folder.")

    st.write("")

    with st.container(border=True):
        st.markdown("### 🔁 One scan, doing the job of two")
        st.markdown(
            "**Normally:** a chest CT checks lungs/heart, and a *separate* "
            "DXA scan (different machine, different visit) checks bone density.\n\n"
            "**With BoneSight-CT:** the *same* chest CT — already being done "
            "for another reason — is reused to also screen for bone density. "
            "One scan a patient already had now does the work of two, at no "
            "extra radiation and no extra cost."
        )

    st.write("")

    with st.container(border=True):
        st.markdown("### 📖 Quick terms")
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown("**DXA scan**")
            st.caption(
                "The gold-standard, low-radiation scan that measures bone "
                "density directly and confirms osteoporosis/osteopenia."
            )
        with t2:
            st.markdown("**Osteoporosis**")
            st.caption(
                "Bone has lost density/strength faster than it can rebuild — "
                "porous, fragile bone with a much higher fracture risk."
            )
        with t3:
            st.markdown("**Osteopenia**")
            st.caption(
                "Lower-than-normal bone density — an early warning stage, "
                "not yet osteoporosis."
            )

    st.write("")

    intro_left, intro_right = st.columns([1.6, 1], gap="large")

    with intro_left:
        with st.container(border=True):
            st.markdown("### Hi, I'm Zainab 👋")
            st.write(
                "I'm a Medical Imaging Technology student interested in how "
                "AI can extend the value of scans that are already being done."
            )
            st.write("This project began with a simple question:")
            st.markdown(
                "**Can we screen for bone density using a chest CT that a "
                "patient is already having for another reason?**"
            )
            st.write(
                "The goal is not to replace DXA. It is to flag people who "
                "may benefit from one, in places where DXA isn't available."
            )

    with intro_right:
        with st.container(border=True):
            st.markdown("### What this prototype does")
            st.write("🦴 Reads an L1 vertebra HU value")
            st.write("📊 Classifies it as Normal / Osteopenia / Osteoporosis")
            st.write("🧪 Shows real sample cases from the dataset")
            st.write("📈 Summarizes results across screened patients")

    st.write("")
    st.markdown("### Start here")

    option1, option2 = st.columns(2, gap="large")

    with option1:
        with st.container(border=True):
            st.markdown("### 🔢 Try the Classifier")
            st.write("Enter an L1 HU value and see the classification instantly.")
            if st.button("Try It Yourself →", type="primary", use_container_width=True):
                navigate("Analyze")

    with option2:
        with st.container(border=True):
            st.markdown("### 🧪 See Sample Cases")
            st.write("Explore three real patients — one from each category.")
            if st.button("View Demo Cases →", use_container_width=True):
                navigate("Demos")

    st.write("")

    with st.container(border=True):
        st.markdown("### Why this matters")
        st.write(
            "Many smaller cities and lower-resource hospitals — including "
            "many in Pakistan — have no dedicated DXA machine at all, while "
            "chest CT is far more widely available. Opportunistic screening "
            "from an existing scan could mean the difference between "
            "osteoporosis being caught early and it not being caught at all."
        )
        st.write(
            "This project draws on my own clinical internship observations: "
            "government-hospital equipment gaps, long queues, and patients "
            "in smaller cities having no local access to imaging beyond a "
            "basic CT or X-ray."
        )

    st.write("")
    st.info(
        "Research/student prototype only. This tool is not clinically "
        "validated and must not be used to diagnose patients or replace a "
        "formal DXA scan."
    )


# ============================================================
# ANALYZE / TRY IT YOURSELF
# ============================================================

elif st.session_state.page == "Analyze":

    st.markdown("### TRY THE CLASSIFIER")
    st.title("Enter an L1 HU Value")

    st.write(
        "Enter a mean Hounsfield Unit (HU) value for the L1 vertebra "
        "(e.g. from a radiology report or research measurement) to see "
        "the Pickhardt-based classification."
    )

    uploaded_scan = st.file_uploader(
        "Optional: upload a CT slice image to view alongside your result "
        "(e.g. an overlay image generated by the BoneSight-CT Colab pipeline)",
        type=["png", "jpg", "jpeg"],
    )

    hu_input = st.slider("Mean L1 HU value", min_value=0, max_value=300, value=150, step=1)

    category, icon, color = classify_hu(hu_input)
    info = interpretation(category)

    left, right = st.columns([1, 1.6], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("##### CLASSIFICATION")
            st.markdown(f"## {icon} {category}")
            st.markdown('<div class="score-label">MEAN L1 HU</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="score-number">{hu_input}</div>', unsafe_allow_html=True)

    with right:
        st.markdown("### What does this mean?")
        st.write(f"**{category}:** {info['definition']}")
        st.write(info["message"])
        st.markdown(f"**Suggested next step:** {info['action']}")
        st.caption(
            "Thresholds (Pickhardt et al., Radiology 2019, validated against "
            "DXA in over 20,000 patients): HU ≤ 110 → Osteoporosis · "
            "110–160 → Osteopenia · > 160 → Normal."
        )
        st.info(DXA_BLURB)

    if uploaded_scan is not None:
        st.divider()
        st.markdown("### Uploaded scan")
        st.image(uploaded_scan, use_container_width=True)
        st.caption(
            "This image is shown for reference only — the classification "
            "above comes from the HU value you entered, not from analyzing "
            "this image live (full segmentation runs offline in Colab; see Learn)."
        )
        st.caption(
            f"Reminder: **{category}** here means {info['definition'].lower()} "
            "A formal DXA scan is the only way to confirm this from an actual scan."
        )

    st.divider()
    st.warning(
        "This is a screening flag, not a diagnosis. Anyone flagged "
        "Osteopenia or Osteoporosis should be referred for a formal DXA scan."
    )


# ============================================================
# DEMO / SAMPLE CASES
# ============================================================

elif st.session_state.page == "Demos":

    st.markdown("### SAMPLE CASES")
    st.title("Real Patients From the Dataset")

    st.write(
        "These three cases are real, de-identified patients from the "
        "public TCIA Lung-PET-CT-Dx dataset used in this project — one "
        "example from each classification category."
    )

    with st.container(border=True):
        st.markdown("##### Quick reference")
        leg1, leg2, leg3 = st.columns(3)
        with leg1:
            st.markdown("🟢 **Normal**")
            st.caption("Healthy bone density by this method.")
        with leg2:
            st.markdown("🟠 **Osteopenia**")
            st.caption("Lower-than-normal density — an early warning stage.")
        with leg3:
            st.markdown("🔴 **Osteoporosis**")
            st.caption("Porous, fragile bone with higher fracture risk.")
        st.caption(DXA_BLURB)

    cols = st.columns(3, gap="medium")

    for i, case in enumerate(DEMO_CASES):
        category, icon, color = classify_hu(case["mean_hu"])

        with cols[i]:
            with st.container(border=True):
                st.markdown(f"**{case['label']}**")
                st.caption(f"Patient ID: {case['id']}")

                if os.path.exists(case["image"]):
                    st.image(
                        case["image"],
                        caption="Original CT (left) vs. L1 vertebra highlighted (right)",
                        use_container_width=True,
                    )
                else:
                    st.info("Overlay image not yet added to sample_images/ folder.")

                st.markdown(f"### {icon} {category}")
                st.metric("Mean L1 HU", f"{case['mean_hu']:.1f}")

                if st.button("See explanation", key=f"demo_{i}", use_container_width=True):
                    info = interpretation(category)
                    st.write(f"**{category}:** {info['definition']}")
                    st.info(info["message"])
                    st.caption(DXA_BLURB)

    st.divider()
    st.caption(
        "Note: this prototype classifies from an already-measured HU value. "
        "Live segmentation (TotalSegmentator) is too computationally heavy "
        "to run inside this deployed app, so it runs offline in Google Colab; "
        "these three cases show its real output."
    )


# ============================================================
# RESULTS
# ============================================================

elif st.session_state.page == "Results":

    st.markdown("### DATASET RESULTS")
    st.title("Full Screening Results")

    if not DATA_READY:
        st.error(f"{RESULTS_CSV} not found in the app folder.")
    else:
        st.markdown("### Classification breakdown")
        counts = success_df["classification"].value_counts()
        st.bar_chart(counts)
        st.caption(
            "🟢 Normal = healthy density · 🟠 Osteopenia = early warning stage · "
            "🔴 Osteoporosis = fragile bone, higher fracture risk. "
            "A formal DXA scan confirms any of these."
        )

        st.markdown("### HU value by patient")
        st.bar_chart(success_df.set_index("patient")["mean_hu"])

        st.markdown("### Full patient table")
        st.dataframe(
            success_df[["patient", "mean_hu", "median_hu", "classification"]],
            use_container_width=True,
            hide_index=True,
        )

        csv_data = success_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download results (CSV)",
            data=csv_data,
            file_name="bonesight_results.csv",
            mime="text/csv",
        )

        st.divider()
        st.markdown("### Pipeline success rate")
        status_counts = df["status"].value_counts()
        st.bar_chart(status_counts)
        st.caption(
            f"{len(success_df)} of {len(df)} attempted patients "
            f"({len(success_df)/len(df)*100:.0f}%) produced a usable L1 HU "
            "measurement. See Learn for why the rest didn't."
        )


# ============================================================
# LEARN
# ============================================================

elif st.session_state.page == "Learn":

    st.markdown("### PROJECT GUIDE")
    st.title("How BoneSight-CT Works")

    with st.container(border=True):
        st.markdown("### 🔁 The core concept: one modality, two jobs")
        st.write(
            "Normally, checking bone density requires a dedicated DXA scan — "
            "a separate machine, separate appointment, separate cost. "
            "BoneSight-CT's core idea is that a chest CT, already being done "
            "for an unrelated reason, contains enough information (the L1 "
            "vertebra) to also produce a bone-density screening signal — "
            "so one imaging modality does the work normally split across two."
        )

    with st.expander("🏥 What is a DXA scan, and why is it done?"):
        st.write(
            "DXA (Dual-energy X-ray Absorptiometry) is the clinical "
            "gold-standard test for measuring bone density. It sends two "
            "X-ray beams of different energy levels through bone (usually "
            "the hip and lower spine); the difference in how much each "
            "beam is absorbed lets a scanner calculate Bone Mineral "
            "Density (BMD) very precisely."
        )
        st.write(
            "It's done to catch bone loss early, estimate fracture risk, "
            "diagnose osteoporosis before a fracture happens, and monitor "
            "how a patient responds to treatment. The scan itself is "
            "quick (10–15 minutes), painless, and uses very low radiation "
            "— much lower than a standard CT."
        )
        st.write(
            "DXA results are usually reported as a **T-score**: a normal "
            "T-score is -1.0 or above, -1.0 to -2.5 is Osteopenia, and "
            "below -2.5 is Osteoporosis. BoneSight-CT doesn't produce a "
            "T-score directly — it uses a different, CT-based measurement "
            "(HU) that has been separately validated against DXA results "
            "in large studies."
        )
        st.write(
            "**The catch:** DXA machines are expensive and far less "
            "common than CT scanners, especially outside major cities — "
            "which is exactly the gap BoneSight-CT is trying to help close."
        )

    with st.expander("🦴 What is Osteoporosis?"):
        st.write(
            "Osteoporosis is a condition where bone tissue loses density "
            "and strength faster than the body can rebuild it. Bones "
            "become porous and fragile — like a sponge with larger holes "
            "— which makes them much more likely to fracture, even from a "
            "minor fall or bump. It's often called a 'silent disease' "
            "because there are usually no symptoms until a fracture occurs."
        )

    with st.expander("🦴 What is Osteopenia?"):
        st.write(
            "Osteopenia is an earlier, milder stage of bone density loss "
            "— lower than normal, but not low enough to be classified as "
            "osteoporosis. It doesn't always progress to osteoporosis, but "
            "it's a meaningful warning sign, and catching it early gives "
            "the most room for lifestyle changes or treatment to help."
        )

    with st.expander("👥 Who gets osteoporosis, and why?"):
        st.write(
            "Osteoporosis becomes far more common with age, because bone "
            "naturally remodels more slowly over time. It is especially "
            "common in **postmenopausal women** — the drop in estrogen "
            "after menopause significantly speeds up bone loss — and in "
            "**men and women over roughly 65–70**."
        )
        st.write(
            "Other risk factors include: family history of osteoporosis "
            "or fractures, low calcium or vitamin D intake, a sedentary "
            "lifestyle, smoking and heavy alcohol use, low body weight, "
            "and long-term use of certain medications (such as steroids)."
        )
        st.write(
            "This is part of why opportunistic screening matters: many of "
            "the people most at risk (older patients) are also the ones "
            "already getting chest CTs for other age-related conditions — "
            "so the opportunity to screen is already there, in a scan "
            "they're already having."
        )

    with st.expander("🩻 Why chest CT?"):
        st.write(
            "Chest CTs are extremely common — lung cancer screening, "
            "pneumonia follow-up, cancer staging — and the L1 vertebra is "
            "often captured as a byproduct of the standard field of view. "
            "No extra scan or radiation is needed."
        )

    with st.expander("🦴 Why the L1 vertebra?"):
        st.write(
            "L1 is the vertebra most consistently included in standard "
            "chest CT scans, and it's the same vertebra validated against "
            "DXA in the Pickhardt et al. (2019) study of over 20,000 patients."
        )

    with st.expander("🔀 Image fusion: the same principle used in PET-CT"):
        st.write(
            "In radiology, **image fusion** (or multimodal fusion) combines "
            "two scans so each covers the other's weakness — most commonly "
            "**PET-CT**, where a PET scan's functional/metabolic signal "
            "(e.g. 'this area is metabolically active') is overlaid onto a "
            "CT scan's precise anatomical detail (exact location, shape, "
            "structure). Alone, PET shows *that* something is happening "
            "but not clearly *where*; CT shows *where* everything is but "
            "not function. Fused together, both questions get answered "
            "in one image."
        )
        st.write(
            "The sample overlay images in the Demos tab use this same "
            "visualization principle, at a smaller scale: instead of "
            "fusing two separate scans, they fuse a single CT scan with "
            "**AI-derived information about that same scan** — the "
            "TotalSegmentator-predicted L1 location — overlaid in color on "
            "top of the grayscale anatomy. It's the same underlying idea "
            "that makes PET-CT fusion useful: don't just report a number, "
            "show *where on the actual scan* that number came from, so a "
            "clinician can visually verify it."
        )

    with st.expander("🔬 What is TotalSegmentator?"):
        st.write(
            "TotalSegmentator is a free, pre-trained AI model that "
            "automatically locates and outlines organs and bones — "
            "including the L1 vertebra — in a CT scan. Using it meant no "
            "custom segmentation model had to be trained from scratch."
        )

    with st.expander("📊 What is a Hounsfield Unit (HU)?"):
        st.write(
            "HU is the standard brightness scale used in CT imaging, "
            "reflecting tissue density. Denser bone produces a higher HU "
            "value; less dense (more porous) bone produces a lower one."
        )

    with st.expander("🧮 The full pipeline, step by step"):
        st.markdown(
            """
            1. **Chest CT acquisition** — a routine scan done for an unrelated reason
            2. **DICOM → NIfTI conversion**
            3. **L1 vertebra segmentation** via TotalSegmentator
            4. **Trabecular bone isolation** — the vertebra mask is eroded by 3mm
               to exclude the dense outer (cortical) shell, leaving the
               spongy trabecular bone most sensitive to early density loss
            5. **Mean HU calculation** inside that trabecular region
            6. **Classification** against the Pickhardt et al. (2019) thresholds
            """
        )

    with st.expander("🎓 What inspired this project?"):
        st.write(
            "BoneSight-CT was inspired by a Taipei Medical University "
            "study (Kuo et al., *International Journal of Medical "
            "Informatics*, 2025), which used a deep-learning (ViT-CNN) "
            "model to recommend DXA follow-up scans directly from chest "
            "low-dose CT images. Dr. Yi-Tien Li is a co-author on that "
            "paper."
        )
        st.write(
            "This project doesn't attempt to reproduce that model — it "
            "uses a simpler, fully transparent HU-threshold method "
            "(Pickhardt et al., 2019) instead of a custom deep-learning "
            "model, since it was built solo, without a mentor or "
            "institutional compute/data access, on a compressed timeline."
        )
        st.write(
            "What it keeps **authentic** to that inspiration is the "
            "underlying idea: a routine chest CT can carry a legitimate, "
            "clinically-grounded bone-density signal, if the right region "
            "is measured correctly. Building this end-to-end — from raw "
            "DICOM data to a working classification pipeline — was about "
            "proving that idea is achievable independently, using "
            "published, peer-reviewed thresholds rather than an unverified "
            "shortcut."
        )

    with st.expander("🧭 How would this work in a real hospital, in the future?"):
        st.markdown(
            """
            This prototype classifies one HU value at a time. A realistic
            future clinical workflow would look like this:

            1. **Automatic trigger** — every chest CT sent to the hospital's
               imaging system (PACS) is automatically checked for whether
               L1 is inside the scanned range.
            2. **Background processing** — if it is, the L1-HU pipeline runs
               quietly in the background (no extra scan, no patient wait).
            3. **Report addendum** — if the result falls in the Osteopenia
               or Osteoporosis range, a flag is added to the radiologist's
               report as a *suggestion*, not an automatic diagnosis.
            4. **Radiologist review** — the radiologist decides, using their
               judgment and the patient's history, whether to recommend a
               formal DXA scan.
            5. **Referral, especially where DXA is scarce** — in hospitals
               without a DXA machine, this flag becomes even more valuable:
               it can prompt a referral to a facility that has one, instead
               of bone loss going undetected entirely.

            The tool's role stays limited on purpose: it surfaces a signal
            for a qualified clinician to act on — it never diagnoses or
            replaces DXA itself.
            """
        )

    with st.expander("🎯 Why build this at all — what's the point?"):
        st.write(
            "The purpose isn't to replace DXA or radiologists. It's to "
            "close a real accessibility gap: chest CT scanners are far "
            "more widely available than DXA machines, especially in "
            "smaller cities and lower-resource hospitals. Every chest CT "
            "that already includes L1 is a missed opportunity for a free "
            "bone-density signal if nobody looks at it that way."
        )
        st.write(
            "This project exists to show — honestly, with real data and "
            "real limitations reported — that a validated, published "
            "method (not a black-box model) can be built end-to-end by a "
            "single student, on public data, without a hospital or mentor, "
            "and still produce a genuinely useful, clinically-grounded "
            "screening signal."
        )

    with st.expander("⚠️ Honest limitations"):
        st.write(
            "- **No DXA ground truth**: this dataset has no matched DXA "
            "scans, so classification relies on published thresholds, not "
            "verification against gold-standard measurements on these exact patients."
        )
        st.write(
            "- **~10% technical success rate**: of 200 patients attempted, "
            "only 20 produced a usable measurement — mostly because L1 fell "
            "outside the scan range, or a known DICOM-conversion library bug "
            "unrelated to actual scan quality."
        )
        st.write(
            "- **Simple threshold approach**: the inspiring TMU paper "
            "(Kuo et al. 2025) uses a more advanced ViT-CNN model; this "
            "project intentionally uses a simpler, transparent threshold "
            "method suited to a solo, no-mentor student timeline."
        )
        st.write(
            "- **Small sample size**: 20 patients demonstrates feasibility, "
            "not clinical validity."
        )

    st.divider()
    st.markdown("### References")
    st.markdown(
        "- Pickhardt, P.J., et al. *Automated CT-based Opportunistic "
        "Osteoporosis Screening.* Radiology (2019)."
    )
    st.markdown(
        "- Kuo, C.Y., et al. *Deep learning chest LDCT to DXA "
        "recommendation.* International Journal of Medical Informatics (2025)."
    )
    st.markdown(
        "- World Health Organization. *WHO Criteria for the Diagnosis of "
        "Osteoporosis* (T-score classification, based on DXA BMD measurement)."
    )
    st.markdown(
        "- International Osteoporosis Foundation — patient-facing "
        "reference on DXA scanning, osteoporosis, and osteopenia."
    )

    st.divider()
    st.markdown("### About this project")
    st.write(
        "Built independently by a Medical Imaging Technology student, "
        "without institutional data access, using the public "
        "Lung-PET-CT-Dx dataset (The Cancer Imaging Archive)."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "BoneSight-CT  •  Zainab Fatima  •  Medical Imaging Technology  •  "
    "Educational / research prototype"
)
