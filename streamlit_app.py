import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.express as px
import pandas as pd
from datetime import date, timedelta
import io
from PIL import Image, ImageDraw, ImageFont

# --- 1. CONFIGURATION AND CONSTANTS ---

st.set_page_config(layout="wide", page_title="Quantum Risk Analysis")

# Assumed statutory rates for April 2025. These must be verified annually.
STAT_WEEKLY_PAY_CAP = 700
STAT_MAX_COMP_AWARD = 115115
STAT_MAX_BASIC_AWARD = 21000 # (30 * 700)
MAX_SERVICE_YEARS = 20
NET_PAY_FACTOR_DEFAULT = 0.70 # Assumes approx. 30% for tax/NI

# Vento Bands for claims presented on or after 6 April 2025 (Assumed)
VENTO_BANDS = {
    "Lower Band (£1,200 - £12,000)": (1200, 12000),
    "Middle Band (£12,000 - £35,900)": (12000, 35900),
    "Upper Band (£35,900 - £59,900)": (35900, 59900),
}

# JCG data for psychiatric injury and chronic pain, per user request.
JCG_PSYCHIATRIC_DATA = {
    "--- Select Injury Bracket ---": (0, 0),
    "Chapter 4: Psychiatric Damage Generally - (a) Severe": (66920, 141240),
    "Chapter 4: Psychiatric Damage Generally - (b) Moderately Severe": (23270, 66920),
    "Chapter 4: Psychiatric Damage Generally - (c) Moderate": (7150, 23270),
    "Chapter 4: Psychiatric Damage Generally - (d) Less Severe": (1880, 7150),
    "Chapter 4: PTSD - (a) Severe": (73050, 122850),
    "Chapter 4: PTSD - (b) Moderately Severe": (28250, 73050),
    "Chapter 4: PTSD - (c) Moderate": (9980, 28250),
    "Chapter 4: PTSD - (d) Less Severe": (4820, 9980),
    "Chapter 9: Other Pain Disorders - (a) Severe (e.g. Fibromyalgia)": (51410, 76870),
    "Chapter 9: Other Pain Disorders - (b) Moderate": (25710, 46970),
}


# --- 2. CORE CALCULATION & HELPER FUNCTIONS ---

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
    """Calculates simple interest on an annual basis."""
    if not isinstance(start_date, date) or not isinstance(end_date, date) or end_date <= start_date:
        return 0
    duration_years = (end_date - start_date).days / 365.25
    return principal_amount * rate * duration_years

def create_export_image(stats_data, chart_fig):
    """Combines stats and a chart into a single PNG image for export."""
    # --- Create Stats Image ---
    export_width = 800
    stats_height = 160
    
    img = Image.new('RGB', (export_width, stats_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Use a default font for maximum portability, with fallback
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        title_font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()
        title_font = font

    # Layout dimensions
    pad = 15
    line_h = 20
    col1, col2, col3, col4 = pad, 275, 500, 650

    draw.text((pad, pad), "Quantum Analysis Summary", fill="black", font=title_font)
    draw.line((pad, pad + line_h + 5, export_width - pad, pad + line_h + 5), fill="grey")

    y_pos = pad + line_h + 15
    draw.text((col1, y_pos), "Mean (Average)", fill="dimgray", font=font)
    draw.text((col1, y_pos + line_h), f"£{stats_data['mean']:,.0f}", fill="black", font=font)
    draw.text((col2, y_pos), "Median (50%)", fill="dimgray", font=font)
    draw.text((col2, y_pos + line_h), f"£{stats_data['50%']:,.0f}", fill="black", font=font)
    draw.text((col3, y_pos), "Std. Deviation", fill="dimgray", font=font)
    draw.text((col3, y_pos + line_h), f"£{stats_data['std']:,.0f}", fill="black", font=font)

    y_pos += 60
    draw.line((pad, y_pos - 5, export_width - pad, y_pos - 5), fill="lightgrey")
    draw.text((col1, y_pos), "10th Percentile", fill="dimgray", font=font)
    draw.text((col1, y_pos + line_h), f"£{stats_data['10%']:,.0f}", fill="black", font=font)
    draw.text((col2, y_pos), "25th Percentile", fill="dimgray", font=font)
    draw.text((col2, y_pos + line_h), f"£{stats_data['25%']:,.0f}", fill="black", font=font)
    draw.text((col3, y_pos), "75th Percentile", fill="dimgray", font=font)
    draw.text((col3, y_pos + line_h), f"£{stats_data['75%']:,.0f}", fill="black", font=font)
    draw.text((col4, y_pos), "90th Percentile", fill="dimgray", font=font)
    draw.text((col4, y_pos + line_h), f"£{stats_data['90%']:,.0f}", fill="black", font=font)
    
    stats_image = img

    # --- Get Chart Image ---
    # The 'scale' parameter is removed to ensure the width is exactly as specified.
    chart_bytes = chart_fig.to_image(format="png", width=export_width, height=400)
    chart_image = Image.open(io.BytesIO(chart_bytes))

    # --- Combine Images ---
    total_height = stats_image.height + chart_image.height
    final_image = Image.new('RGB', (export_width, total_height), 'white')
    final_image.paste(stats_image, (0, 0))
    final_image.paste(chart_image, (0, stats_image.height))

    # --- Convert to bytes for download ---
    img_byte_arr = io.BytesIO()
    final_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


# --- 3. STREAMLIT UI AND SIMULATION LOGIC ---
st.title("⚖️ Employment & PI Quantum: Monte Carlo Analysis Tool")

# --- LEGAL DISCLAIMER ---
st.warning(
    """
    **For Educational & Illustrative Purposes Only**

    - **This is not legal advice.** The output is a statistical estimation based on the variables you input. It is not a substitute for a formal opinion from a qualified barrister or solicitor on the specific facts of a case.
    - Use of this tool **does not create a barrister-client relationship**.
    - The figures generated are for informational purposes only and **should not be relied upon** for making any legal, financial, or strategic decisions.
    - The accuracy of the output is entirely dependent on the assumptions and data entered. No warranty is given as to its accuracy.
    - **No liability is accepted** for any loss or damage arising from the use of this tool. For professional advice, please seek a formal opinion.
    """
)

st.markdown("---")

# --- SIDEBAR FOR USER INPUTS ---
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    n_simulations = st.select_slider(
        "Number of Scenarios to Run",
        options=[1000, 5000, 10000, 20000, 50000],
        value=10000
    )
    today = date.today()
    trial_date = st.date_input("Assumed Trial Date", value=today + timedelta(days=365))
    st.info("Define the parameters for each head of claim you wish to include in the analysis.")

    with st.expander("1. Unfair Dismissal", expanded=False):
        include_ud = st.checkbox("Include Unfair Dismissal Claim", key="ud_check")
        st.subheader("Claimant Details")
        c1, c2 = st.columns(2)
        age_at_dismissal = c1.number_input("Age at Dismissal", 16, 80, 45)
        service_years = c2.number_input("Continuous Service (Yrs)", 2, 40, 10)
        st.subheader("Weekly Gross Income (£)")
        c1, c2, c3 = st.columns(3)
        ud_income_min = c1.number_input("Min", value=500, key="ud_inc_min")
        ud_income_mode = c2.number_input("Most Likely", value=750, key="ud_inc_mode")
        ud_income_max = c3.number_input("Max", value=1000, key="ud_inc_max")
        st.subheader("Future Loss Period (Months)")
        c1, c2, c3 = st.columns(3)
        ud_loss_min = c1.number_input("Min", value=6.0, key="ud_loss_min", format="%.1f")
        ud_loss_mode = c2.number_input("Most Likely", value=9.0, key="ud_loss_mode", format="%.1f")
        ud_loss_max = c3.number_input("Max", value=12.0, key="ud_loss_max", format="%.1f")
        st.subheader("Adjustments & Caps")
        net_pay_factor = st.slider("Net Pay as % of Gross", 1, 100, int(NET_PAY_FACTOR_DEFAULT * 100)) / 100
        polkey_reduction = st.slider("Polkey / Contrib. Reduction (%)", 0, 100, 0) / 100
        acas_uplift = st.slider("ACAS Code Uplift (%)", 0, 25, 0) / 100
        disapply_comp_cap = st.checkbox("Disapply compensatory award cap", help="Select if the dismissal is automatically unfair (e.g., for whistleblowing) or discriminatory.")
    
    with st.expander("2. Discrimination (Injury to Feelings)", expanded=False):
        include_disc = st.checkbox("Include Discrimination Claim", key="disc_check")
        selected_band_key = st.selectbox("Select Vento Band", VENTO_BANDS.keys())
        st.subheader("Award within Vento Band (£)")
        min_val, max_val = VENTO_BANDS[selected_band_key]
        default_mode = round(min_val + (max_val - min_val) * 0.25)
        c1, c2, c3 = st.columns(3)
        disc_award_min = c1.number_input("Min", value=min_val, key="disc_min")
        disc_award_mode = c2.number_input("Most Likely", value=default_mode, key="disc_mode")
        disc_award_max = c3.number_input("Max", value=max_val, key="disc_max")
        st.subheader("Interest on Injury to Feelings")
        apply_disc_interest = st.checkbox("Apply 8% Interest", key="disc_int_check")
        interest_start_date_disc = st.date_input("Interest Start Date", value=today, key="disc_int_start")

    with st.expander("3. Personal Injury (Psychiatric)", expanded=False):
        include_pi = st.checkbox("Include Personal Injury Claim", key="pi_check")
        selected_pi_key = st.selectbox("Select JCG Bracket", JCG_PSYCHIATRIC_DATA.keys())
        st.subheader("Award within JCG Bracket (£)")
        pi_min_val, pi_max_val = JCG_PSYCHIATRIC_DATA.get(selected_pi_key, (0, 0))
        pi_default_mode = round(pi_min_val + (pi_max_val - pi_min_val) * 0.5)
        c1, c2, c3 = st.columns(3)
        pi_award_min = c1.number_input("Min", value=pi_min_val, key="pi_min")
        pi_award_mode = c2.number_input("Most Likely", value=pi_default_mode, key="pi_mode")
        pi_award_max = c3.number_input("Max", value=pi_max_val, key="pi_max")
        st.subheader("Interest on Personal Injury")
        apply_pi_interest = st.checkbox("Apply 8% Interest", key="pi_int_check")
        interest_start_date_pi = st.date_input("Interest Start Date", value=today, key="pi_int_start")

    with st.expander("4. Pension Loss", expanded=False):
        include_pension = st.checkbox("Include Pension Loss", key="pension_check")
        pension_method = st.radio("Calculation Method", ["Simplified (Weekly Loss)", "Capitalised Total Loss"], horizontal=True)
        if pension_method == "Simplified (Weekly Loss)":
            st.subheader("Weekly Pension Loss (£)")
            c1, c2, c3 = st.columns(3)
            pen_loss_min = c1.number_input("Min", value=20.0, key="pen_min", format="%.2f")
            pen_loss_mode = c2.number_input("Most Likely", value=40.0, key="pen_mode", format="%.2f")
            pen_loss_max = c3.number_input("Max", value=60.0, key="pen_max", format="%.2f")
        else:
            st.subheader("Total Capitalised Pension Loss (£)")
            c1, c2, c3 = st.columns(3)
            pen_loss_min = c1.number_input("Min", value=5000, key="pen_cap_min")
            pen_loss_mode = c2.number_input("Most Likely", value=10000, key="pen_cap_mode")
            pen_loss_max = c3.number_input("Max", value=15000, key="pen_cap_max")

    run_button = st.button("🚀 Run Simulation")

# --- MAIN PANEL FOR RESULTS ---
if run_button:
    total_awards = np.zeros(n_simulations)
    if include_ud:
        age_factor = np.clip(age_at_dismissal - 21, 0, 20) * 0.5 + np.clip(age_at_dismissal - 40, 0, 20) * 0.5
        capped_service = min(service_years, MAX_SERVICE_YEARS)
        capped_weekly_pay = min(ud_income_mode, STAT_WEEKLY_PAY_CAP)
        basic_award = min(age_factor * capped_weekly_pay * capped_service, STAT_MAX_BASIC_AWARD)
        gross_pay_samples = pert_sample(ud_income_min, ud_income_mode, ud_income_max, n_simulations)
        loss_period_samples = pert_sample(ud_loss_min, ud_loss_mode, ud_loss_max, n_simulations)
        loss_in_weeks = loss_period_samples * (365.25 / 12 / 7)
        compensatory_award = (gross_pay_samples * net_pay_factor) * loss_in_weeks
        if not disapply_comp_cap:
            cap_1 = STAT_MAX_COMP_AWARD
            cap_2 = gross_pay_samples * 52
            compensatory_award = np.minimum(compensatory_award, np.minimum(cap_1, cap_2))
        compensatory_award *= (1 - polkey_reduction) * (1 + acas_uplift)
        total_awards += (basic_award + compensatory_award)
    if include_disc:
        disc_award_samples = pert_sample(disc_award_min, disc_award_mode, disc_award_max, n_simulations)
        if apply_disc_interest:
            disc_award_samples += calculate_interest(disc_award_samples, interest_start_date_disc, trial_date)
        total_awards += disc_award_samples
    if include_pi and selected_pi_key != "--- Select Injury Bracket ---":
        pi_award_samples = pert_sample(pi_award_min, pi_award_mode, pi_award_max, n_simulations)
        if apply_pi_interest:
            pi_award_samples += calculate_interest(pi_award_samples, interest_start_date_pi, trial_date)
        total_awards += pi_award_samples
    if include_pension:
        pension_loss_samples = pert_sample(pen_loss_min, pen_loss_mode, pen_loss_max, n_simulations)
        pension_loss_total = np.zeros(n_simulations)
        if pension_method == "Simplified (Weekly Loss)":
            if 'loss_in_weeks' in locals():
                pension_loss_total = pension_loss_samples * loss_in_weeks
            else:
                st.warning("Simplified pension loss needs a UD loss period. Using Capitalised Loss method instead.")
                pension_loss_total = pension_loss_samples
        else:
            pension_loss_total = pension_loss_samples
        total_awards += pension_loss_total

    st.header("📊 Simulation Results")
    if np.sum(total_awards) == 0:
        st.warning("No claims were included or all values were zero. Please select at least one claim type and run again.")
    else:
        results_df = pd.DataFrame(total_awards, columns=['Total Award'])
        stats = results_df['Total Award'].describe(percentiles=[.10, .25, .50, .75, .90])
        st.subheader("Summary Statistics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean (Average) Award", f"£{stats['mean']:,.0f}")
        c2.metric("Median (50th Percentile)", f"£{stats['50%']:,.0f}")
        c3.metric("Standard Deviation", f"£{stats['std']:,.0f}")
        st.subheader("Percentile Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("10th Percentile", f"£{stats['10%']:,.0f}", help="10% of outcomes are below this value.")
        c2.metric("25th Percentile", f"£{stats['25%']:,.0f}", help="25% of outcomes are below this value.")
        c3.metric("75th Percentile", f"£{stats['75%']:,.0f}", help="75% of outcomes are below this value.")
        c4.metric("90th Percentile", f"£{stats['90%']:,.0f}", help="90% of outcomes are below this value.")
        st.markdown("---")
        st.subheader("Distribution of Potential Outcomes")
        fig = px.histogram(results_df, x='Total Award', nbins=100, title='Frequency Distribution of Total Award Value', labels={'Total Award': 'Total Award Value (£)'}, histnorm='probability density')
        fig.update_layout(yaxis_title="Probability Density", bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

        # --- EXPORT BUTTON LOGIC ---
        try:
            image_bytes = create_export_image(stats, fig)
            st.download_button(
                label="📥 Export Results as PNG",
                data=image_bytes,
                file_name=f"quantum_analysis_{date.today()}.png",
                mime="image/png"
            )
        except ImportError:
             st.warning("Could not generate export image. Please install Pillow: pip install Pillow")
        except Exception as e:
            st.warning(f"Could not generate export image. Error: {e}")
            st.warning("This may be due to missing font files (e.g., 'arial.ttf') on the system. The tool attempts to use a basic default font if specific ones are not found.")

else:
    st.info("⬅️ Please configure your claim parameters in the sidebar and click 'Run Simulation' to see the results.")
