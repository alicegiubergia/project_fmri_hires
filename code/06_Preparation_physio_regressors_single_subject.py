#!/usr/bin/env python3
"""
Preprocess physiological recordings (cardiac and respiratory)
and compute physiological regressors (HRV, RVT, RV) in BIDS-derivatives format.

Input:  BIDS-compliant physio files in `bids_dataset/sub-XX/func/`
Output: derivatives/physio/sub-XX/func/tmp_run-XX/{*_peaks.phys, *_HRV.tsv, *_RVT.tsv, *_RV.tsv, *_plots.html}
"""

import os
import numpy as np
import nibabel as nib
import peakdet as pk
import plotly.graph_objects as go
import plotly.io as pio
from phys2denoise.metrics import (
    heart_rate_variability,
    respiratory_variance,
    respiratory_variance_time,
    retroicor
)
from phys2denoise.metrics.utils import export_metric
import matplotlib
matplotlib.use('TkAgg')  # Use non-interactive backend for matplotlib

# ============================================
# CONFIGURATION
# ============================================

BIDS_DIR = "/mnt/store2/projects/progetto_layer/data"
DERIV_DIR = "/mnt/store2/projects/progetto_layer/data/derivatives/physio" 

subject = "sub-008"
runs = ["01", "02", "03"]
task = "reading"

FS_CARD = 100  # Hz
FS_RESP = 25   # Hz

def calculate_slice_times(N, TR, vendor='GE'):
    slice_times = np.zeros(N)
    delta_t = TR / N
    
    if vendor == 'siemens':
        if N % 2 != 0: # DISPARI
            order = list(range(0, N, 2)) + list(range(1, N, 2))
        else:          # PARI
            order = list(range(1, N, 2)) + list(range(0, N, 2))
    else: # GE/Philips standard interleaved
        order = list(range(0, N, 2)) + list(range(1, N, 2))
        
    for i, slice_idx in enumerate(order):
        slice_times[slice_idx] = i * delta_t
        
    return slice_times

# ============================================
# MAIN LOOP
# ============================================

for run in runs:
    print(f"\n==============================")
    print(f"🧠 Processing {subject}, run-{run}")
    print(f"==============================")

    func_dir = os.path.join(BIDS_DIR, subject, "func")
    work_dir = os.path.join(DERIV_DIR, subject, "func", f"tmp_run-{run}")
    os.makedirs(work_dir, exist_ok=True)

    card_file = os.path.join(func_dir, f"{subject}_task-{task}_run-{run}_recording-cardiac_physio.tsv")
    resp_file = os.path.join(func_dir, f"{subject}_task-{task}_run-{run}_recording-respiratory_physio.tsv")
    nifti_file = os.path.join(func_dir, f"{subject}_task-{task}_run-{run}_part-mag_bold.nii.gz")

    if not os.path.exists(nifti_file):
        print(f"⚠️ Skipping run-{run}: NIfTI not found.")
        continue
    if not os.path.exists(card_file) or not os.path.exists(resp_file):
        print(f"⚠️ Skipping run-{run}: missing physio file(s).")
        continue

    img = nib.load(nifti_file)
    n_tp = img.shape[3]
    TR = float(img.header.get_zooms()[3])
    n_slices = img.shape[2]
    print(f"→ TR = {TR:.3f} s | n_tp = {n_tp}")

    # ============================================
    # CARDIAC SIGNAL
    # ============================================
    base_card = os.path.basename(card_file).replace("_physio.tsv", "")
    outfile_card = os.path.join(work_dir, f"{base_card}_peaks.phys")
    """
    print(f"\n=== Processing Cardiac ===")
    data_card = np.genfromtxt(card_file)
    phys_card = pk.Physio(data_card[1:], fs=FS_CARD)

    # Filter
    phys_card_filt = pk.operations.filter_physio(phys_card, cutoffs=1.5, method="lowpass", order=7)
    # Peak detection
    phys_card_peaks = pk.operations.peakfind_physio(phys_card_filt, thresh=0.3, dist=None)
    phys_card_peaks = pk.operations.edit_physio(phys_card_peaks)


    pk.save_physio(outfile_card, phys_card_peaks)
    print(f"✅ Saved: {outfile_card}")

    # --- Plot pre/post cardiac ---
    t_card = np.arange(len(phys_card.data)) / FS_CARD
    fig_card = go.Figure()
    fig_card.add_trace(go.Scatter(x=t_card, y=phys_card.data, mode='lines', name='Raw', line=dict(color='gray', dash='dot', width=.6)))
    fig_card.add_trace(go.Scatter(x=t_card, y=phys_card_filt.data, mode='lines', name='Filtered', line=dict(color='royalblue')))
    fig_card.add_trace(go.Scatter(x=t_card[phys_card_peaks.peaks], y=phys_card_filt.data[phys_card_peaks.peaks],
                                  mode='markers', name='Peaks', marker=dict(color='red', size=6)))
    fig_card.update_layout(title=f"Cardiac signal (run-{run})", xaxis_title="Time [s]", yaxis_title="Amplitude", template="plotly_white")
    fig_card.show(renderer="browser")
    """

    # ============================================
    # RESPIRATORY SIGNAL
    # ============================================

    base_resp = os.path.basename(resp_file).replace("_physio.tsv", "")
    outfile_resp = os.path.join(work_dir, f"{base_resp}_peaks.phys")
    """
    print(f"\n=== Processing Respiratory ===")
    data_resp = np.genfromtxt(resp_file)
    phys_resp = pk.Physio(data_resp[1:], fs=FS_RESP)
    phys_resp_filt = pk.operations.filter_physio(phys_resp, cutoffs=.5, method="lowpass", order=7)
    phys_resp_peaks = pk.operations.peakfind_physio(phys_resp_filt, thresh=0.1, dist=None)
    phys_resp_peaks = pk.operations.edit_physio(phys_resp_peaks)

    pk.save_physio(outfile_resp, phys_resp_peaks)
    print(f"✅ Saved: {outfile_resp}")

    # --- Plot pre/post respiratory ---
    t_resp = np.arange(len(phys_resp.data)) / FS_RESP
    fig_resp = go.Figure()
    fig_resp.add_trace(go.Scatter(x=t_resp, y=phys_resp.data, mode='lines', name='Raw', line=dict(color='gray', dash='dot', width=.6)))
    fig_resp.add_trace(go.Scatter(x=t_resp, y=phys_resp_filt.data, mode='lines', name='Filtered', line=dict(color='#00b5f7')))
    fig_resp.add_trace(go.Scatter(x=t_resp[phys_resp_peaks.peaks], y=phys_resp_filt.data[phys_resp_peaks.peaks],
                                  mode='markers', name='Peaks', marker=dict(color='red', size=6)))
    fig_resp.add_trace(go.Scatter(x=t_resp[phys_resp_peaks.troughs], y=phys_resp_filt.data[phys_resp_peaks.troughs],
                                mode='markers', name='Troughs', marker=dict(color='green', size=6)))
    fig_resp.update_layout(title=f"Respiratory signal (run-{run})", xaxis_title="Time [s]", yaxis_title="Amplitude", template="plotly_white")
    fig_resp.show(renderer="browser")
    """
    
    # ============================================
    # HRV computation 
    # ============================================

    card_prefix = os.path.join(work_dir, f"{base_card}_HRV")
    card = pk.load_physio(outfile_card, allow_pickle=True)

    # --------- (Frequency domain [Hz]) ----------

    HRV = heart_rate_variability(card.data, card.peaks, card.fs, window=6)
    export_metric(HRV[:,1], card.fs, tr=TR, fileprefix=card_prefix, ntp=n_tp, ext=".csv")
    print(f"💾 Exported HRV → {card_prefix}*")

    fig_hrv = go.Figure()
    fig_hrv.add_trace(go.Scatter(y=HRV[:,0], mode='lines', name='raw HRV', line=dict(color='gray', dash='dot', width=.6 )))

    fig_hrv.add_trace(go.Scatter(y=HRV[:,1], mode='lines', name='HRV ∗ CRF', line=dict(color='purple', width=.8)))
    fig_hrv.update_layout(title=f"HRV (run-{run})", xaxis_title="Samples", yaxis_title="HRV", template="plotly_white")
    fig_hrv.show(renderer="browser")

    # --------- (Time domain [s]) ----------

    peak_times = card.peaks / card.fs  # Convert peak indices to time in seconds
    rri_s = np.diff(peak_times)  # R-R intervals in seconds
    rri_ms = rri_s * 1000  # Convert to milliseconds

    # === Compute stats ===
    mean_rri = np.mean(rri_ms)
    std_rri = np.std(rri_ms)
    median_rri = np.median(rri_ms)
    mad_rri = np.median(np.abs(rri_ms - median_rri))


    fig_hrv_time = go.Figure()
    # Histogram of R-R intervals
    fig_hrv_time.add_trace(go.Histogram(x=rri_ms, nbinsx=40, name='R-R Intervals', marker_color='cornflowerblue', opacity=1))

    # === Shaded areas ===
    # Mean ± 2*SD
    fig_hrv_time.add_vrect(
        x0=mean_rri - 2*std_rri, x1=mean_rri + 2*std_rri, fillcolor="plum", opacity=0.15, layer="below", line_width=0, annotation_text="Mean ± 2SD", annotation_position="top left")

    # === Vertical lines ===
    fig_hrv_time.add_vline(x=mean_rri, line=dict(color='plum', width=2), annotation_text='Mean RRi', annotation_position='top right')

    # === Layout ===
    fig_hrv_time.update_layout(
        title=f"Distribution of RR intervals (run-{run})",
        xaxis_title="RR interval [ms]",
        yaxis_title="Count",
        template="simple_white",
        bargap=0,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1
        )
    )

    fig_hrv_time.show(renderer="browser")


    # ============================================
    # RVT computation
    # ============================================
    resp_prefix_rvt = os.path.join(work_dir, f"{base_resp}_RVT")
    resp = pk.load_physio(outfile_resp, allow_pickle=True)
    RVT = respiratory_variance_time(resp.data, resp.peaks, resp.troughs, resp.fs, lags=(0, 4, 8, 12))
    export_metric(RVT[:,1], resp.fs, tr=TR, fileprefix=resp_prefix_rvt, ntp=n_tp)
    print(f"💾 Exported RVT → {resp_prefix_rvt}*")

    fig_rvt = go.Figure()
    fig_rvt.add_trace(go.Scatter(y=RVT[:,0], mode='lines', name='raw RVT', line=dict(color='gray', dash='dot', width=.6 )))
    fig_rvt.add_trace(go.Scatter(y=RVT[:,1], mode='lines', name='RVT ∗ CRF', line=dict(color='darkorange', width=.8)))
    fig_rvt.update_layout(title=f"RVT (run-{run})", xaxis_title="Samples", yaxis_title="RVT", template="plotly_white")

    # ============================================
    # RV computation
    # ============================================
    resp_prefix_rv = os.path.join(work_dir, f"{base_resp}_RV")
    RV = respiratory_variance(resp.data, resp.fs)
    export_metric(RV[:,1], resp.fs, tr=TR, fileprefix=resp_prefix_rv, ntp=n_tp, ext=".csv")
    print(f"💾 Exported RV → {resp_prefix_rv}*")
    
    # ============================================
    # RETROICOR computation
    # ============================================
    """
    ### da implementare perche codice under development in phys2denoise, 
    times = calculate_slice_times(n_slices, TR, vendor='siemens')

    resp_prefix_retroicor = os.path.join(deriv_dir, f"{base_resp}_RETROICOR")
    RETROICOR = retroicor(resp.data, t_r=TR, n_scans=n_tp, slice_timings=times , n_harmonics=2, resp=True, card=False)

    export_metric(RETROICOR, resp.fs, tr=TR, fileprefix=resp_prefix_retroicor, ntp=n_tp)
    print(f"💾 Exported Respiratory RETROICOR → {resp_prefix_retroicor}*" )

    card_prefix_retroicor = os.path.join(deriv_dir, f"{base_card}_RETROICOR")
    RETROICOR = retroicor(card.data, t_r=TR, n_scans=n_tp, slice_timings=times , n_harmonics=2, resp=False, card=True)

    export_metric(RETROICOR, card.fs, tr=TR, fileprefix=card_prefix_retroicor, ntp=n_tp)
    print(f"💾 Exported Cardiac RETROICOR → {card_prefix_retroicor}*" )
    """####


    # Pick chosen files for output in the main folder

    suffix = 'resampled'  # Suffix for HRV, RVT, RV files after resampling and convolution
    for metric in [('cardiac', 'HRV'), ('respiratory', 'RV')]:
        src_file = os.path.join(work_dir, f"{base_card}_{metric[1]}_{suffix}.csv") if metric[1] == 'HRV' else os.path.join(work_dir, f"{base_resp}_{metric[1]}_{suffix}.csv")
        dst_file = os.path.join(DERIV_DIR, subject, "func", f"{subject}_task-{task}_run-{run}_recording-{metric[0]}_{metric[1]}_{suffix}.csv")
        if os.path.exists(src_file):
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            os.replace(src_file, dst_file)
            print(f"📁 Moved {metric} to main folder: {dst_file}")
        else:
            print(f"⚠️ {metric} file not found for run-{run}.")



    fig_rv = go.Figure()
    fig_rv.add_trace(go.Scatter(y=RV[:,0], mode='lines', name='raw RV', line=dict(color='gray', dash='dot', width=.6 )))
    fig_rv.add_trace(go.Scatter(y=RV[:,1], mode='lines', name='RV ∗ CRF', line=dict(color='green', width=.8)))
    fig_rv.update_layout(title=f"RV (run-{run})", xaxis_title="Samples", yaxis_title="RV", template="plotly_white")

    print(f"✅ Finished {subject} run-{run}\n")
    plot=False
    if plot==True:
        figs = [fig_card, fig_hrv, fig_hrv_time, fig_resp, fig_rvt, fig_rv]

        sections = [
            ("💓 Cardiac Signal", "Segnale cardiaco grezzo e filtrato con i picchi R rilevati.", fig_card),
            ("🌬️ Respiratory Signal", "Segnale respiratorio grezzo e filtrato con i picchi e le troughs rilevati.", fig_resp),
            ("💓 HRV", "Variabilità della frequenza cardiaca (HRV).", fig_hrv),
            ("📈 R-R Intervals", "Distribuzione degli intervalli R-R.", fig_hrv_time),
            ("🌬️⏱️ RVT", "Variabilità della frequenza respiratoria (RVT).", fig_rvt),
            ("🌬️📊 RV", "Varianza respiratoria (RV).", fig_rv),
        ]

        html_plots = []
        for title, desc, fig in sections:
            # Genera HTML dei plot con script Plotly incluso dal CDN
            fig_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False,
                                default_height=400, default_width=800)
            html_plots.append(f"<h2>{title}</h2><p>{desc}</p>{fig_html}")

        final_html = f"""
        <html>
        <head>
        <meta charset="utf-8">
        <title>{subject}_run-{run}</title>
        <style>
            body {{
            font-family: "Apple Color Emoji", "Segoe UI Emoji", Arial, sans-serif;
            margin: 40px;
            }}
            h1 {{
            color: #1a1a1a;
            }}
            h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 5px;
            margin-top: 50px;
            }}
            p {{
            color: #555;
            margin-bottom: 15px;
            }}
            .plotly-graph-div {{
            width: 100% !important;
            min-height: 400px !important;
            }}
        </style>
        </head>
        <body>
        <h1>Physiological Report – {subject} run-{run}</h1>
        {''.join(html_plots)}
        </body>
        </html>
        """

        output_path = f"{work_dir}/report_run-{run}_report.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_html)

        print(f"✅ Report HTML salvato in: {output_path}")



print("\n🎯 All runs processed successfully.")


