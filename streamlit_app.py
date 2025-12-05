import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.express as px
import pandas as pd
from datetime import date, timedelta
import io
from PIL import Image, ImageDraw, ImageFont

# --- 1. CONFIGURATION AND STYLING ---
st.set_page_config(layout="wide", page_title="Quantum Risk Analysis")

# Hide Streamlit elements to look like a native website tool
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. CONSTANTS (2025 RATES) ---
STAT_WEEKLY_PAY_CAP = 700
STAT_MAX_COMP_AWARD = 115115
STAT_MAX_BASIC_AWARD = 21000 
MAX_SERVICE_YEARS = 20
NET_PAY_FACTOR_DEFAULT = 0.70 

VENTO_BANDS = {
    "Lower Band (£1,200 - £12,000)": (1200, 12000),
    "Middle Band (£12,000 - £35,900)": (12000, 35900),
    "Upper Band (£35,900 - £59,900)": (35900, 59900),
}

JCG_PSYCHIATRIC_DATA = {
    "--- Select Injury Bracket ---": (0, 0),
    "Psychiatric: Severe (£66k - £141k)": (66920, 141240),
    "Psychiatric: Mod. Severe (£23k - £66k)": (23270, 66920),
    "Psychiatric: Moderate (£7k - £23k)": (7150, 23270),
    "Psychiatric: Less Severe (£1.8k - £7k)": (1880, 7150),
    "PTSD: Severe (£73k - £122k)": (73050, 122850),
    "PTSD: Mod. Severe (£28k - £73k)": (28250, 73050),
    "PTSD: Moderate (£9.9k - £28k)": (9980, 28250),
    "PTSD: Less Severe (£4.8k - £9.9k)": (4820, 9980),
}

# --- 3. HELPER FUNCTIONS ---

def pert_sample(min_val, most_likely, max_val, n, gamma=4):
    """Generates a sample from a PERT (modified Beta) distribution."""
    if not (min_val <= most_likely <= max_val):
        return np.random.uniform(min_val, max_val, n)
    if min_val == max_val:
        return np.full(n, min_val)

    epsilon = 1e-12
    alpha = 1 + gamma * (most_likely - min_val) / (max_val - min_val + epsilon)
    beta = 1 + gamma * (max_val - most_likely) / (max_val - min_val + epsilon)
    
    return min_val + stats.beta.rvs(alpha, beta, size=n) * (max_val - min_val)

def calculate_interest(principal_amount, start_date, end_date, rate=0.08):
    if not isinstance(start_date, date) or not isinstance(end_date, date) or end_date <= start_date:
        return 0
    duration_years = (end_date - start_date).days / 365.25
    return principal_amount * rate * duration_years

def create_export_image(stats_data, chart_fig):
    """Combines stats and a chart into a single PNG image for export."""
    export_width = 800
    stats_height = 180
    
    img = Image.new('RGB', (export_width, stats_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Robust Font Loading for Linux/Cloud Environments
    try:
        # Try a standard font if available
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except IOError:
        # Fallback to default bitmap font
        font = ImageFont.load_default()
        title_font = font

    pad = 20
    line_h = 25
    col1, col2, col3, col4 = pad, 250, 450, 650

    # Draw Text
    draw.text((pad, pad), "QUANTUM RISK ANALYSIS SUMMARY", fill="black", font=title_font)
    draw.line((pad, pad + 30, export_width - pad, pad + 30), fill="black", width=2)

    y_pos = pad + 45
    draw.text((col1, y_pos), "Mean Award", fill="dimgray", font=font)
    draw.text((col1, y_pos + line_h), f"£{stats_data['mean']:,.0f}", fill="black", font=font)
    draw.text((col2, y_pos), "Median (50%)", fill="dimgray", font=font)
    draw.text((col2, y_pos + line_h), f"£{stats_data['50%']:,.0f}", fill="black", font=font)
    draw.text((col3, y_pos), "Std. Deviation", fill="dimgray", font=font)
    draw.text((col3, y_pos + line_h), f"£{stats_data['std']:,.0f}", fill="black", font=font)

    y_pos += 70
    draw.line((pad, y_pos - 10, export_width - pad, y_pos - 10), fill="lightgrey")
    draw.text((col1, y_pos), "10th %ile", fill="dimgray", font=font)
    draw.text((col1, y_pos + line_h), f"£{stats_data['10%']:,.0f}", fill="black", font=font)
    draw.text((col2, y_pos), "25th %ile", fill="dimgray", font=font)
    draw.text((col2, y_pos + line_h), f"£{stats_data['25%']:,.0f}", fill="black", font=font)
    draw.text((col3, y_pos), "75th %ile", fill="dimgray", font=font)
    draw.text((col3, y_pos + line_h), f"£{stats_data['75%']:,.0f}", fill="black", font=font)
    draw.text((col4, y_pos), "90th %ile", fill="dimgray", font=font)
    draw.text((col4, y_pos + line_h), f"£{stats_data['90%']:,.0f}", fill="black", font=font)
    
    stats_image = img

    # Get Chart Image
    chart_bytes = chart_fig.to_image(format="png", width=export_width, height=400, scale=2)
    chart_image = Image.open(io.BytesIO(chart_bytes))

    # Combine
    total_height = stats_image.height + chart_image.height
    final_image = Image.new('RGB', (export_width, total_height), 'white')
    final_image.paste(stats_image, (0, 0))
    final_image.paste(chart_image, (0, stats_image.height))

    img_byte_arr = io.BytesIO()
    final_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


# --- 4. UI & LOGIC ---
st.title("⚖️ Employment Quantum Risk Estimator")
st.markdown("Monte Carlo simulation using **PERT distributions** to model litigation risk and uncertainty.")

# SIDEBAR
with st.sidebar:
    st.header("1. Settings")
    n_simulations = st.select_slider("Simulations", [1000, 5000, 10000, 20000], value=5000)
    trial_date = st.date_input("Trial Date", value=date.today() + timedelta(days=365))

    st.header("2. Heads of Claim")
    
    # UNFAIR DISMISSAL
    with st.expander("Unfair Dismissal", expanded=True):
        include_ud = st.checkbox("Include UD Claim", value=True)
        if include_ud:
            c1, c2 = st.columns(2)
            age = c1.number_input("Age", 16, 80, 45)
            service = c2.number_input("Service (Yrs)", 2, 40, 10)
            
            st.caption("Net Weekly Loss (£)")
            ud_min = st.number_input("Min", 0, 5000, 400, key="u1")
            ud_mode = st.number_input("Likely", 0, 5000, 600, key="u2")
            ud_max = st.number_input("Max", 0, 5000, 700, key="u3")
            
            st.caption("Future Loss (Months)")
            loss_min = st.number_input("Min Months", 0.0, 36.0, 6.0)
            loss_mode = st.number_input("Likely Months", 0.0, 36.0, 9.0)
            loss_max = st.number_input("Max Months", 0.0, 36.0, 12.0)
            
            polkey = st.slider("Polkey / Contrib. Deduction (%)", 0, 100, 0)
            uplift = st.slider("ACAS Uplift (%)", 0, 25, 0)
            no_cap = st.checkbox("Disapply Cap (Whistleblowing/Discrim)")

    # DISCRIMINATION
    with st.expander("Injury to Feelings (Vento)"):
        include_disc = st.checkbox("Include Injury to Feelings")
        if include_disc:
            band = st.selectbox("Vento Band", VENTO_BANDS.keys(), index=1)
            v_min, v_max = VENTO_BANDS[band]
            v_mode = int(v_min + (v_max - v_min) * 0.25)
            
            st.caption("Award Range (£)")
            d_min = st.number_input("Min", value=v_min, key="d1")
            d_mode = st.number_input("Likely", value=v_mode, key="d2")
            d_max = st.number_input("Max", value=v_max, key="d3")
            
            apply_int_disc = st.checkbox("Add 8% Interest", value=True, key="int_d")

    # PERSONAL INJURY
    with st.expander("Personal Injury (JCG)"):
        include_pi = st.checkbox("Include PSLA")
        if include_pi:
            pi_cat = st.selectbox("Category", JCG_PSYCHIATRIC_DATA.keys())
            p_min_v, p_max_v = JCG_PSYCHIATRIC_DATA[pi_cat]
            p_mode_v = int(p_min_v + (p_max_v - p_min_v) * 0.5)
            
            st.caption("Award Range (£)")
            pi_min = st.number_input("Min", value=p_min_v, key="p1")
            pi_mode = st.number_input("Likely", value=p_mode_v, key="p2")
            pi_max = st.number_input("Max", value=p_max_v, key="p3")
            
            apply_int_pi = st.checkbox("Add 8% Interest", value=True, key="int_p")

    # PENSION
    with st.expander("Pension Loss"):
        include_pen = st.checkbox("Include Pension")
        if include_pen:
            st.caption("Total Capitalised Loss (£)")
            pl_min = st.number_input("Min", value=5000, key="pl1")
            pl_mode = st.number_input("Likely", value=8000, key="pl2")
            pl_max = st.number_input("Max", value=12000, key="pl3")

    st.markdown("---")
    run_btn = st.button("🚀 Run Simulation", type="primary")

# CALCULATION ENGINE
if run_btn:
    total_awards = np.zeros(n_simulations)
    
    # 1. Unfair Dismissal
    if include_ud:
        age_fac = np.clip(age - 21, 0, 20) * 0.5 + np.clip(age - 40, 0, 20) * 0.5
        basic = min(age_fac * min(ud_mode, STAT_WEEKLY_PAY_CAP) * min(service, MAX_SERVICE_YEARS), STAT_MAX_BASIC_AWARD)
        
        wk_pay = pert_sample(ud_min, ud_mode, ud_max, n_simulations)
        months = pert_sample(loss_min, loss_mode, loss_max, n_simulations)
        
        comp = wk_pay * (months * 4.345) # Convert months to weeks approx
        
        # Statutory Cap Logic
        if not no_cap:
            stat_cap = STAT_MAX_COMP_AWARD
            gross_cap = wk_pay * 52 # Crude approx using net as proxy if gross not supplied, safe for conservative est
            comp = np.minimum(comp, np.minimum(stat_cap, gross_cap))
            
        comp = comp * (1 - (polkey/100)) * (1 + (uplift/100))
        total_awards += (basic + comp)

    # 2. Discrimination
    if include_disc:
        d_val = pert_sample(d_min, d_mode, d_max, n_simulations)
        if apply_int_disc:
            d_val += calculate_interest(d_val, date.today(), trial_date)
        total_awards += d_val

    # 3. PI
    if include_pi:
        p_val = pert_sample(pi_min, pi_mode, pi_max, n_simulations)
        if apply_int_pi:
            p_val += calculate_interest(p_val, date.today(), trial_date)
        total_awards += p_val

    # 4. Pension
    if include_pen:
        pen_val = pert_sample(pl_min, pl_mode, pl_max, n_simulations)
        total_awards += pen_val

    # RESULTS DISPLAY
    if np.sum(total_awards) == 0:
        st.error("Total award is zero. Please select claims.")
    else:
        df = pd.DataFrame(total_awards, columns=['Award'])
        stats = df['Award'].describe(percentiles=[.10, .25, .50, .75, .90])

        # Metrics
        st.subheader("Results Analysis")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Award", f"£{stats['mean']:,.0f}")
        c2.metric("Median (50%)", f"£{stats['50%']:,.0f}")
        c3.metric("Standard Deviation", f"£{stats['std']:,.0f}")

        # Probability Table
        st.markdown("#### Probability Thresholds")
        col_probs = st.columns(4)
        col_probs[0].metric("Conservative (10%)", f"£{stats['10%']:,.0f}", help="90% chance of exceeding this")
        col_probs[1].metric("Cautious (25%)", f"£{stats['25%']:,.0f}")
        col_probs[2].metric("Optimistic (75%)", f"£{stats['75%']:,.0f}")
        col_probs[3].metric("Upper End (90%)", f"£{stats['90%']:,.0f}", help="Only 10% chance of exceeding this")

        # Chart
        st.markdown("---")
        fig = px.histogram(
            df, x='Award', nbins=60, 
            title='Likelihood Distribution',
            color_discrete_sequence=['#333333'], # Dark grey for professional look
            template="simple_white"
        )
        fig.update_layout(
            xaxis_title="Total Award Value (£)", 
            yaxis_title="Scenario Count",
            bargap=0.1
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export
        try:
            img_bytes = create_export_image(stats, fig)
            st.download_button("📥 Download Report Image", img_bytes, "quantum_risk.png", "image/png")
        except Exception as e:
            st.error(f"Export Error: {e}")
