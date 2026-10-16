import numpy as np
import matplotlib
import os
import numpy as np
import plotly.graph_objs as go

matplotlib.use('TkAgg')

def plot_physio_with_triggers(data_file, trig_file, fs, color, title):
    """
    Plot physiological signal with fMRI trigger markers (interactive Plotly).
    
    Parameters:
        data_file (str): path to signal file (e.g., PPGData_*)
        trig_file (str): path to trigger file (e.g., PPGTrig_*)
        fs (float): sampling frequency in Hz
        color (str): color for the signal line (e.g., 'teal' or 'blue')
        title (str): plot title
    """
    # === Load data ===
    data = np.loadtxt(data_file)
    trig = np.loadtxt(trig_file)

    # --- time vectors ---
    t_data = np.arange(len(data)) / fs
    trigger_times = trig / fs  # if trig are samples; comment if already in sec

    # === Create figure ===
    fig = go.Figure()

    # --- main signal ---
    fig.add_trace(go.Scatter(
        x=t_data,
        y=data,
        mode='lines',
        name='Physio signal',
        line=dict(color=color)
    ))

    # --- add trigger lines ---
    for tt in trigger_times:
        fig.add_vline(
            x=tt,
            line=dict(color='orange', width=1, dash='dot'),
            opacity=0.7
        )

    # === Layout ===
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Signal amplitude",
        template="plotly_white",
        autosize=True,
        xaxis=dict(
            range=[0, 1000 / fs],   # mostra i primi 1000 punti
            rangeslider=dict(visible=True),  # permette lo scroll
        )
    )

    #fig.show()
    fig.show(renderer="browser")  # se esegui da terminale

# --- Set working directory ---
os.chdir('/mnt/store2/projects/progetto_layer/sourcedata/4688/gating')

# --- 1. PPG ---
plot_physio_with_triggers(
    data_file="PPGData_epi_0721202512_56_28_291",
    trig_file="PPGTrig_epi_0721202512_56_28_291",
    fs=100,  # Hz
    color='red',
    title="PPG signal"
)

# --- 2. RESP ---
plot_physio_with_triggers(
    data_file="RESPData_epi_0721202512_56_28_291",
    trig_file="RESPTrig_epi_0721202512_56_28_291",
    fs=25,  # Hz
    color='blue',
    title="Respiratory signal"
)
