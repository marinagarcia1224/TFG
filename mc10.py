# ============================================================================================
# 1. LIBRERÍAS Y CONFIGURACIÓN INICIAL
# ============================================================================================
import os
import traceback
import numpy as np
import awkward as ak
import uproot
import vector
import atlasopenmagic as atom
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator
from lmfit import Model
from lmfit.models import GaussianModel, BreitWignerModel

# --- Configuración de ATLAS Open Data ---
print("Configurando entorno y buscando archivos Monte Carlo (MC)...")
atom.set_release('2025e-13tev-beta')
skim = "GamGam"

# --- Configuración del análisis ---
fraction_mc = 1.0
plot_dir = "plots_mc10"
os.makedirs(plot_dir, exist_ok=True)

mc_datasets = {
    "ggH": ['343981'], 
    "VBF": ['346214'], 
    "VH":  ['345317', '345318', '345319', '345061'], 
    "ttH": ['346525'],
    "yy_bkg": ['302521'] 
}

variables = [
    "photon_pt", "photon_eta", "photon_phi", "photon_e", "photon_isTightID", "photon_ptcone20",
    "jet_pt", "jet_eta", "jet_phi", "jet_e", "jet_btag_quantile", 
    "lep_n", "lep_pt", "lep_eta", "lep_phi", "lep_e",
    "mcWeight"
]

# --- Descarga de URLs ---
mc_files = {cat: [] for cat in mc_datasets.keys()}
print("Descargando URLs de las simulaciones...")
for categoria, dsids in mc_datasets.items():
    for dsid in dsids:
        urls = atom.get_urls(dsid, skim, protocol='https', cache=True)
        mc_files[categoria].extend(urls)
    print(f" ✓ {categoria}: {len(mc_files[categoria])} archivos listos.")


# ============================================================================================
# 2. FUNCIONES DE SELECCIÓN Y CINEMÁTICA
# ============================================================================================

# --- Cortes de fotones ---
def cut_photon_reconstruction(photon_isTightID):
    return (photon_isTightID[:,0]==True) & (photon_isTightID[:,1]==True)

def cut_photon_pt(photon_pt):
    return (photon_pt[:,0] > 50) & (photon_pt[:,1] > 30)

def cut_isolation_pt(photon_ptcone20, photon_pt):
    return ((photon_ptcone20[:,0]/photon_pt[:,0]) < 0.055) & ((photon_ptcone20[:,1]/photon_pt[:,1]) < 0.055)

def cut_photon_eta_transition(photon_eta):
    condition_0 = (np.abs(photon_eta[:, 0]) < 1.37) | (np.abs(photon_eta[:, 0]) > 1.52)
    condition_1 = (np.abs(photon_eta[:, 1]) < 1.37) | (np.abs(photon_eta[:, 1]) > 1.52)
    return condition_0 & condition_1

def calc_mass(photon_pt, photon_eta, photon_phi, photon_e):
    p4 = vector.zip({"pt": photon_pt, "eta": photon_eta, "phi": photon_phi, "e": photon_e})
    return (p4[:, 0] + p4[:, 1]).M

def cut_iso_mass(photon_pt, invariant_mass):
    return ((photon_pt[:,0]/invariant_mass) > 0.35) & ((photon_pt[:,1]/invariant_mass) > 0.35)

# --- Variables de Jets y Leptones ---
def get_clean_jets_mask(jet_pt):
    return (jet_pt > 25) 

def count_bjets(jet_btag_quantile, clean_mask):
    is_bjet = (jet_btag_quantile >= 3) & clean_mask
    return ak.sum(is_bjet, axis=1)

def calc_mjj(jet_pt, jet_eta, jet_phi, jet_e):
    p4 = vector.zip({"pt": jet_pt, "eta": jet_eta, "phi": jet_phi, "e": jet_e})
    p4_padded = ak.pad_none(p4, 2, axis=1)
    return ak.fill_none((p4_padded[:, 0] + p4_padded[:, 1]).M, 0.0)

def calc_mll(lep_pt, lep_eta, lep_phi, lep_e):
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "e": lep_e})
    p4_padded = ak.pad_none(p4, 2, axis=1)
    return ak.fill_none((p4_padded[:, 0] + p4_padded[:, 1]).M, 0.0)

def calc_mlgamma_masses(lep_pt, lep_eta, lep_phi, lep_e, ph_pt, ph_eta, ph_phi, ph_e):
    lep_p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "e": lep_e})
    ph_p4 = vector.zip({"pt": ph_pt, "eta": ph_eta, "phi": ph_phi, "e": ph_e})
    lep1 = ak.pad_none(lep_p4, 1, axis=1)[:, 0]
    ph_padded = ak.pad_none(ph_p4, 2, axis=1)
    m_lgamma1 = ak.fill_none((lep1 + ph_padded[:, 0]).M, 0.0)
    m_lgamma2 = ak.fill_none((lep1 + ph_padded[:, 1]).M, 0.0)
    return m_lgamma1, m_lgamma2

def calc_eta_gg(ph_pt, ph_eta, ph_phi, ph_e):
    p4 = vector.zip({"pt": ph_pt, "eta": ph_eta, "phi": ph_phi, "e": ph_e})
    return (p4[:, 0] + p4[:, 1]).eta

def calc_zeppenfeld(eta_gg, eta_j1, eta_j2):
    return np.abs(eta_gg - 0.5 * (eta_j1 + eta_j2))

# --- Categorización ---
def is_tth(lep_n, n_jets, n_bjets):
    tth_lep = (lep_n >= 1) & (n_jets >= 2) & (n_bjets >= 1)
    tth_had = (lep_n == 0) & (n_jets >= 3) & (n_bjets >= 1)
    return tth_lep | tth_had

def is_vh(lep_n, n_jets, mjj, m_ll, m_lgamma1, m_lgamma2):
    vh_dilep = (lep_n >= 2) & (m_ll >= 70.0) & (m_ll <= 110.0)
    pass_veto_1 = np.abs(m_lgamma1 - 89.0) > 5.0 
    pass_veto_2 = np.abs(m_lgamma2 - 89.0) > 5.0
    vh_lep = (lep_n == 1) & pass_veto_1 & pass_veto_2
    vh_had = (lep_n == 0) & (n_jets >= 2) & (mjj > 60.0) & (mjj < 120.0)
    return vh_dilep | vh_lep | vh_had

def is_vbf(n_jets, jet_eta, eta_gg, mjj):
    mask_2j = (n_jets >= 2) 
    eta_padded = ak.pad_none(jet_eta, 2, axis=1)
    delta_eta = ak.fill_none(np.abs(eta_padded[:, 0] - eta_padded[:, 1]) > 2, False) 
    z_val = calc_zeppenfeld(eta_gg, eta_padded[:, 0], eta_padded[:, 1]) 
    zeppenfeld = ak.fill_none(z_val < 5, False)
    mass_cut = (mjj > 250.0)
    return mask_2j & delta_eta & zeppenfeld & mass_cut

# --- Funciones  ---
def double_cb_pdf(x, amplitude, center, sigma, beta_L, m_L, beta_R, m_R):
    t = (x - center) / sigma
    a_L = np.abs(beta_L)
    a_R = np.abs(beta_R)
    
    A_L = (m_L / a_L)**m_L * np.exp(-0.5 * a_L**2)
    B_L = m_L / a_L - a_L
    
    A_R = (m_R / a_R)**m_R * np.exp(-0.5 * a_R**2)
    B_R = m_R / a_R - a_R
    
    y = np.zeros_like(t)
    
    core_mask = (t >= -a_L) & (t <= a_R)
    y[core_mask] = np.exp(-0.5 * t[core_mask]**2)
    
    left_mask = t < -a_L
    y[left_mask] = A_L * (B_L - t[left_mask])**(-m_L)
    
    right_mask = t > a_R
    y[right_mask] = A_R * (B_R + t[right_mask])**(-m_R)
    
    return amplitude * y


# ============================================================================================
# 3. PROCESAMIENTO DE LOS DATOS MC
# ============================================================================================

# Estructuras de almacenamiento
base_keys = ["nlep", "njets", "nbjets", "deta", "zepp", "mjj_vbf", "mjj_vh", "mll", "mlg"]
data_plot = {cat: {k: [] for k in base_keys} for cat in mc_datasets.keys()}
data_weights = {cat: {k: [] for k in base_keys} for cat in mc_datasets.keys()}

categorias_verdad = list(mc_datasets.keys()) 
categorias_reco = ["ttH", "VH", "VBF", "ggH"] 

matriz_conteos = {verdad: {reco: 0.0 for reco in categorias_reco} for verdad in categorias_verdad}
mc_masses = {cat: [] for cat in categorias_verdad}
mc_weights_fit = {cat: [] for cat in categorias_verdad}

for proceso, archivos in mc_files.items():
    print(f"\nProcesando simulaciones de: {proceso}...")
    for afile in archivos:
        try:
            tree_mc = uproot.open(afile + ":analysis", timeout=300, num_workers=4)
            for data in tree_mc.iterate(variables, library="ak", entry_stop=tree_mc.num_entries*fraction_mc):
                
                # Cortes base
                data = data[cut_photon_reconstruction(data['photon_isTightID'])]
                data = data[cut_photon_pt(data['photon_pt'])]
                data = data[cut_isolation_pt(data['photon_ptcone20'], data['photon_pt'])]
                data = data[cut_photon_eta_transition(data['photon_eta'])]
                
                mass = calc_mass(data['photon_pt'], data['photon_eta'], data['photon_phi'], data['photon_e'])
                mask_mass = (mass > 100) & (mass < 160)
                data = data[mask_mass]
                mass = mass[mask_mass]
                
                mask_iso = cut_iso_mass(data['photon_pt'], mass)
                data = data[mask_iso]
                mass = mass[mask_iso]

                w = data['mcWeight']

                # Variables para categorización
                clean_mask = get_clean_jets_mask(data['jet_pt'])
                n_jets = ak.sum(clean_mask, axis=1)
                n_bjets = count_bjets(data['jet_btag_quantile'], clean_mask)
                
                mjj_full = calc_mjj(data['jet_pt'], data['jet_eta'], data['jet_phi'], data['jet_e'])
                mll_full = calc_mll(data['lep_pt'], data['lep_eta'], data['lep_phi'], data['lep_e'])
                mlg1_full, mlg2_full = calc_mlgamma_masses(data['lep_pt'], data['lep_eta'], data['lep_phi'], data['lep_e'], data['photon_pt'], data['photon_eta'], data['photon_phi'], data['photon_e'])
                eta_gg_full = calc_eta_gg(data['photon_pt'], data['photon_eta'], data['photon_phi'], data['photon_e'])

                # Almacenamiento 
                data_plot[proceso]["nlep"].append(ak.to_numpy(data['lep_n']))
                data_weights[proceso]["nlep"].append(ak.to_numpy(w))
                
                data_plot[proceso]["njets"].append(ak.to_numpy(n_jets))
                data_weights[proceso]["njets"].append(ak.to_numpy(w))
                
                data_plot[proceso]["nbjets"].append(ak.to_numpy(n_bjets))
                data_weights[proceso]["nbjets"].append(ak.to_numpy(w))

                # Categorización para matriz
                m_tth = is_tth(data['lep_n'], n_jets, n_bjets)
                m_vh = is_vh(data['lep_n'], n_jets, mjj_full, mll_full, mlg1_full, mlg2_full) & ~m_tth
                m_vbf = is_vbf(n_jets, data['jet_eta'], eta_gg_full, mjj_full) & ~m_tth & ~m_vh
                m_ggh = ~m_tth & ~m_vh & ~m_vbf

                matriz_conteos[proceso]["ttH"] += ak.sum(w[m_tth])
                matriz_conteos[proceso]["VH"] += ak.sum(w[m_vh])
                matriz_conteos[proceso]["VBF"] += ak.sum(w[m_vbf])
                matriz_conteos[proceso]["ggH"] += ak.sum(w[m_ggh])

                mc_masses[proceso].append(ak.to_numpy(mass)) 
                mc_weights_fit[proceso].append(ak.to_numpy(w))

                # Variables complejas condicionadas
                mask_2j = (n_jets >= 2)
                if ak.sum(mask_2j) > 0:
                    eta_padded = ak.pad_none(data['jet_eta'], 2, axis=1)
                    eta_j1 = eta_padded[:, 0][mask_2j]
                    eta_j2 = eta_padded[:, 1][mask_2j]
                    
                    data_plot[proceso]["deta"].append(ak.to_numpy(np.abs(eta_j1 - eta_j2)))
                    data_weights[proceso]["deta"].append(ak.to_numpy(w[mask_2j]))
                    
                    data_plot[proceso]["zepp"].append(ak.to_numpy(np.abs(eta_gg_full[mask_2j] - 0.5 * (eta_j1 + eta_j2))))
                    data_weights[proceso]["zepp"].append(ak.to_numpy(w[mask_2j]))
                    
                    data_plot[proceso]["mjj_vbf"].append(ak.to_numpy(mjj_full[mask_2j]))
                    data_weights[proceso]["mjj_vbf"].append(ak.to_numpy(w[mask_2j]))
                    
                    mask_vh_had = mask_2j & (data['lep_n'] == 0)
                    data_plot[proceso]["mjj_vh"].append(ak.to_numpy(mjj_full[mask_vh_had]))
                    data_weights[proceso]["mjj_vh"].append(ak.to_numpy(w[mask_vh_had]))

                mask_2l = (data['lep_n'] >= 2)
                if ak.sum(mask_2l) > 0:
                    data_plot[proceso]["mll"].append(ak.to_numpy(mll_full[mask_2l]))
                    data_weights[proceso]["mll"].append(ak.to_numpy(w[mask_2l]))
                    
                mask_1l = (data['lep_n'] >= 1)
                if ak.sum(mask_1l) > 0:
                    mlg1_filtered = mlg1_full[mask_1l]
                    mlg2_filtered = mlg2_full[mask_1l]
                    w_filtered = w[mask_1l]
                    
                    data_plot[proceso]["mlg"].append(np.concatenate([ak.to_numpy(mlg1_filtered), ak.to_numpy(mlg2_filtered)]))
                    data_weights[proceso]["mlg"].append(np.concatenate([ak.to_numpy(w_filtered), ak.to_numpy(w_filtered)]))

        except Exception as e:
            print(f"  ⚠ Error procesando el archivo {afile}:")
            traceback.print_exc()
            break 

# Concatenar arrays finales
for proc in mc_files.keys():
    for var in data_plot[proc].keys():
        if len(data_plot[proc][var]) > 0:
            data_plot[proc][var] = np.concatenate(data_plot[proc][var])
            data_weights[proc][var] = np.concatenate(data_weights[proc][var])
        else:
            data_plot[proc][var] = np.array([])
            data_weights[proc][var] = np.array([])

for cat in categorias_verdad:
    if len(mc_masses[cat]) > 0:
        mc_masses[cat] = np.concatenate(mc_masses[cat])
        mc_weights_fit[cat] = np.concatenate(mc_weights_fit[cat])
    else:
        mc_masses[cat] = np.array([])
        mc_weights_fit[cat] = np.array([])


# ============================================================================================
# 4. GRÁFICOS DE LAS VARIABLES
# ============================================================================================
print("\nGenerando y guardando los gráficos de forma (normalizados a 1)...")

def get_combined(cats, var):
    d_list = [data_plot[c][var] for c in cats if len(data_plot[c][var]) > 0]
    w_list = [data_weights[c][var] for c in cats if len(data_weights[c][var]) > 0]
    d = np.concatenate(d_list) if d_list else np.array([])
    w = np.concatenate(w_list) if w_list else np.array([])
    return d, w

def plot_custom_stack(ax, lines_data, lines_weights, labels, colors, bins, title, xlabel):
    valid_data, valid_weights, valid_labels, valid_colors = [], [], [], []
    for d, w, l, c in zip(lines_data, lines_weights, labels, colors):
        if len(d) > 0:
            valid_data.append(d)
            valid_weights.append(w)
            valid_labels.append(l)
            valid_colors.append(c)

    if not valid_data:
        return

    ax.hist(valid_data, bins=bins, stacked=False, density=True, histtype='step', 
            label=valid_labels, color=valid_colors, weights=valid_weights, linewidth=2)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Event fraction (normalized)")
    ax.set_yscale('log')
    ax.legend(loc='upper right', frameon=False, fontsize='small')
    
    ax.text(0.95, 0.82, r"$\it{ATLAS}$ Simulation", transform=ax.transAxes,    
            fontsize=12, verticalalignment='top', horizontalalignment='right')

# --- FIGURA 1: Análisis ttH ---
fig1, axs1 = plt.subplots(1, 5, figsize=(30, 5))
mask_tth_lep = data_plot["ttH"]["nlep"] >= 1
mask_tth_had = data_plot["ttH"]["nlep"] == 0

d_oth1, w_oth1 = get_combined(["VH", "VBF", "ggH"], "nlep")
d_bg1, w_bg1 = get_combined(["yy_bkg"], "nlep")
d_tth_lep, w_tth_lep = data_plot["ttH"]["nlep"][mask_tth_lep], data_weights["ttH"]["nlep"][mask_tth_lep]
d_tth_had, w_tth_had = data_plot["ttH"]["nlep"][mask_tth_had], data_weights["ttH"]["nlep"][mask_tth_had]
d_oth2, w_oth2 = get_combined(["VH", "VBF", "ggH"], "njets")
d_bg2, w_bg2 = get_combined(["yy_bkg"], "njets")
d_tth_lep_j, w_tth_lep_j = data_plot["ttH"]["njets"][mask_tth_lep], data_weights["ttH"]["njets"][mask_tth_lep]
d_tth_had_j, w_tth_had_j = data_plot["ttH"]["njets"][mask_tth_had], data_weights["ttH"]["njets"][mask_tth_had]
d_oth3, w_oth3 = get_combined(["VH", "VBF", "ggH"], "nbjets")
d_bg3, w_bg3 = get_combined(["yy_bkg"], "nbjets")
d_tth, w_tth = get_combined(["ttH"], "nbjets")

plot_custom_stack(axs1[0], [d_oth1, d_tth_lep, d_bg1], [w_oth1, w_tth_lep, w_bg1], ["VH+VBF+ggH", "ttH lep", r"$\gamma\gamma$"], ['#d62728', '#1f77b4', '#2ca02c'], np.arange(0, 7)-0.5, "Lepton Multiplicity (ttH lep)", r"$N_{lep}$")
plot_custom_stack(axs1[1], [d_oth1, d_tth_had, d_bg1], [w_oth1, w_tth_had, w_bg1], ["VH+VBF+ggH", "ttH had", r"$\gamma\gamma$"], ['#d62728', '#ff7f0e', '#2ca02c'], np.arange(0, 7)-0.5, "Lepton Multiplicity (ttH had)", r"$N_{lep}$")
plot_custom_stack(axs1[2], [d_oth2, d_tth_lep_j, d_bg2], [w_oth2, w_tth_lep_j, w_bg2], ["VH+VBF+ggH", "ttH lep", r"$\gamma\gamma$"], ['#d62728', '#1f77b4', '#2ca02c'], np.arange(0, 10)-0.5, "Jet Multiplicity (ttH lep)", r"$N_{jets}$")
plot_custom_stack(axs1[3], [d_oth2, d_tth_had_j, d_bg2], [w_oth2, w_tth_had_j, w_bg2], ["VH+VBF+ggH", "ttH had", r"$\gamma\gamma$"], ['#d62728', '#ff7f0e', '#2ca02c'], np.arange(0, 10)-0.5, "Jet Multiplicity (ttH had)", r"$N_{jets}$")
plot_custom_stack(axs1[4], [d_oth3, d_tth, d_bg3], [w_oth3, w_tth, w_bg3], ["VH+VBF+ggH", "ttH", r"$\gamma\gamma$"], ['#d62728', '#8c564b', '#2ca02c'], np.arange(0, 6)-0.5, "b-Jet Multiplicity (ttH mode)", r"$N_{b-jets}$")

plt.tight_layout()
fig1.savefig(os.path.join(plot_dir, "cortes_tth_shapes.png"), dpi=300)

# --- FIGURA 2: Análisis VH ---
fig2, axs2 = plt.subplots(1, 7, figsize=(42, 5))
mask_vh_dilep = data_plot["VH"]["nlep"] >= 2
mask_vh_1lep = data_plot["VH"]["nlep"] == 1
mask_vh_had = data_plot["VH"]["nlep"] == 0

d_oth_vh1, w_oth_vh1 = get_combined(["ttH", "VBF", "ggH"], "nlep")
d_bg_vh1, w_bg_vh1 = get_combined(["yy_bkg"], "nlep")
d_vh_dilep, w_vh_dilep = data_plot["VH"]["nlep"][mask_vh_dilep], data_weights["VH"]["nlep"][mask_vh_dilep]
d_vh_1lep, w_vh_1lep = data_plot["VH"]["nlep"][mask_vh_1lep], data_weights["VH"]["nlep"][mask_vh_1lep]
d_vh_had_l, w_vh_had_l = data_plot["VH"]["nlep"][mask_vh_had], data_weights["VH"]["nlep"][mask_vh_had]

plot_custom_stack(axs2[0], [d_oth_vh1, d_vh_dilep, d_bg_vh1], [w_oth_vh1, w_vh_dilep, w_bg_vh1], ["ttH+VBF+ggH", "VH dilep", r"$\gamma\gamma$"], ['#d62728', '#1f77b4', '#2ca02c'], np.arange(0, 7)-0.5, "Lepton Multiplicity (VH dilep)", r"$N_{lep}$")
plot_custom_stack(axs2[1], [d_oth_vh1, d_vh_1lep, d_bg_vh1], [w_oth_vh1, w_vh_1lep, w_bg_vh1], ["ttH+VBF+ggH", "VH 1lep", r"$\gamma\gamma$"], ['#d62728', '#ff7f0e', '#2ca02c'], np.arange(0, 7)-0.5, "Lepton Multiplicity (VH 1lep)", r"$N_{lep}$")
plot_custom_stack(axs2[2], [d_oth_vh1, d_vh_had_l, d_bg_vh1], [w_oth_vh1, w_vh_had_l, w_bg_vh1], ["ttH+VBF+ggH", "VH had", r"$\gamma\gamma$"], ['#d62728', '#9467bd', '#2ca02c'], np.arange(0, 7)-0.5, "Lepton Multiplicity (VH had)", r"$N_{lep}$")

d_oth_vh2, w_oth_vh2 = get_combined(["ttH", "VBF", "ggH"], "mll")
d_bg_vh2, w_bg_vh2 = get_combined(["yy_bkg"], "mll")
d_vh_mll, w_vh_mll = get_combined(["VH"], "mll")
plot_custom_stack(axs2[3], [d_oth_vh2, d_vh_mll, d_bg_vh2], [w_oth_vh2, w_vh_mll, w_bg_vh2], ["ttH+VBF+ggH", "VH dilep", r"$\gamma\gamma$"], ['#d62728', '#1f77b4', '#2ca02c'], np.linspace(50, 130, 20), "Dilepton Mass", r"$m_{ll}$ [GeV]")

d_oth_vh3, w_oth_vh3 = get_combined(["ttH", "VBF", "ggH"], "mlg")
d_bg_vh3, w_bg_vh3 = get_combined(["yy_bkg"], "mlg")
d_vh_mlg, w_vh_mlg = get_combined(["VH"], "mlg")
plot_custom_stack(axs2[4], [d_oth_vh3, d_vh_mlg, d_bg_vh3], [w_oth_vh3, w_vh_mlg, w_bg_vh3], ["ttH+VBF+ggH", "VH 1lep", r"$\gamma\gamma$"], ['#d62728', '#ff7f0e', '#2ca02c'], np.linspace(50, 130, 15), "Lepton-Photon Mass", r"$m_{l\gamma}$ [GeV]")

d_oth_vh4, w_oth_vh4 = get_combined(["ttH", "VBF", "ggH"], "njets")
d_bg_vh4, w_bg_vh4 = get_combined(["yy_bkg"], "njets")
d_vh_had_j, w_vh_had_j = data_plot["VH"]["njets"][mask_vh_had], data_weights["VH"]["njets"][mask_vh_had]
plot_custom_stack(axs2[5], [d_oth_vh4, d_vh_had_j, d_bg_vh4], [w_oth_vh4, w_vh_had_j, w_bg_vh4], ["ttH+VBF+ggH", "VH had", r"$\gamma\gamma$"], ['#d62728', '#9467bd', '#2ca02c'], np.arange(0, 10)-0.5, "Jet Multiplicity (VH had)", r"$N_{jets}$")

d_oth_vh5, w_oth_vh5 = get_combined(["ttH", "VBF", "ggH"], "mjj_vh")
d_bg_vh5, w_bg_vh5 = get_combined(["yy_bkg"], "mjj_vh")
d_vh_mjj, w_vh_mjj = get_combined(["VH"], "mjj_vh")
plot_custom_stack(axs2[6], [d_oth_vh5, d_vh_mjj, d_bg_vh5], [w_oth_vh5, w_vh_mjj, w_bg_vh5], ["ttH+VBF+ggH", "VH had", r"$\gamma\gamma$"], ['#d62728', '#9467bd', '#2ca02c'], np.linspace(0, 200, 20), "Dijet Mass (VH had)", r"$m_{jj}$ [GeV]")

plt.tight_layout()
fig2.savefig(os.path.join(plot_dir, "cortes_vh_shapes.png"), dpi=300)

# --- FIGURA 3: Análisis VBF ---
fig3, axs3 = plt.subplots(1, 3, figsize=(18, 5))
d_oth_vbf1, w_oth_vbf1 = get_combined(["ttH", "VH", "ggH"], "deta")
d_bg_vbf1, w_bg_vbf1 = get_combined(["yy_bkg"], "deta")
d_vbf1, w_vbf1 = get_combined(["VBF"], "deta")
plot_custom_stack(axs3[0], [d_oth_vbf1, d_vbf1, d_bg_vbf1], [w_oth_vbf1, w_vbf1, w_bg_vbf1], ["ttH+VH+ggH", "VBF", r"$\gamma\gamma$"], ['#d62728', '#1f77b4', '#2ca02c'], np.linspace(0, 6, 20), r"Jet Ang. Sep. ($N_{jets} \geq 2$)", r"$|\Delta\eta_{jj}|$")

d_oth_vbf2, w_oth_vbf2 = get_combined(["ttH", "VH", "ggH"], "zepp")
d_bg_vbf2, w_bg_vbf2 = get_combined(["yy_bkg"], "zepp")
d_vbf2, w_vbf2 = get_combined(["VBF"], "zepp")
plot_custom_stack(axs3[1], [d_oth_vbf2, d_vbf2, d_bg_vbf2], [w_oth_vbf2, w_vbf2, w_bg_vbf2], ["ttH+VH+ggH", "VBF", r"$\gamma\gamma$"], ['#d62728', '#1f77b4', '#2ca02c'], np.linspace(0, 8, 20), "Zeppenfeld Variable", r"$Z_{\gamma\gamma}$")

d_oth_vbf3, w_oth_vbf3 = get_combined(["ttH", "VH", "ggH"], "mjj_vbf")
d_bg_vbf3, w_bg_vbf3 = get_combined(["yy_bkg"], "mjj_vbf")
d_vbf3, w_vbf3 = get_combined(["VBF"], "mjj_vbf")
plot_custom_stack(axs3[2], [d_oth_vbf3, d_vbf3, d_bg_vbf3], [w_oth_vbf3, w_vbf3, w_bg_vbf3], ["ttH+VH+ggH", "VBF", r"$\gamma\gamma$"], ['#d62728', '#1f77b4', '#2ca02c'], np.linspace(0, 1000, 20), "Dijet Invariant Mass", r"$m_{jj}$ [GeV]")

plt.tight_layout()
fig3.savefig(os.path.join(plot_dir, "cortes_vbf_shapes.png"), dpi=300)


# ============================================================================================
# 5. MATRICES DE ACEPTACIÓN, EFICIENCIA Y PUREZA
# ============================================================================================
print("\nGenerando matrices de composición...")

mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Liberation Sans', 'Arial', 'Helvetica']

def draw_atlas_labels(ax):
    y_pos = 1.04 
    ax.text(0.0, y_pos, "ATLAS", transform=ax.transAxes, fontsize=16, fontweight='bold', fontstyle='italic', ha='left')
    ax.text(0.17, y_pos, "Simulation", transform=ax.transAxes, fontsize=16, weight='normal', ha='left')
    ax.text(0.50, y_pos, r"$H \to \gamma\gamma, m_H = 125$ GeV", transform=ax.transAxes, fontsize=14, weight='normal', ha='left')

categorias_verdad_matriz = ["ggH", "VBF", "VH", "ttH"]
matriz_array = np.zeros((len(categorias_verdad_matriz), len(categorias_reco)))
for i, verdad in enumerate(categorias_verdad_matriz):
    for j, reco in enumerate(categorias_reco):
        matriz_array[i, j] = matriz_conteos[verdad][reco]

# --- Matriz de eficiencia ---
sumas_por_fila = matriz_array.sum(axis=1, keepdims=True)
sumas_por_fila[sumas_por_fila == 0] = 1 
matriz_fracciones = (matriz_array / sumas_por_fila) * 100

fig4, ax4 = plt.subplots(figsize=(8, 6.5))
divider4 = make_axes_locatable(ax4)
cax4 = divider4.append_axes("right", size="5%", pad=0.15)
sns.heatmap(matriz_fracciones, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=[f"Reco {c}" for c in categorias_reco],
            yticklabels=[f"True {c}" for c in categorias_verdad_matriz],
            cbar_ax=cax4, cbar_kws={'label': 'Fraction of true events (%)'},
            annot_kws={"size": 11, "weight": "normal"},
            square=True, linewidths=1.0, linecolor='white', ax=ax4)
ax4.set_xlabel("Reconstructed category", weight='normal', labelpad=12)
ax4.set_ylabel("True production mode", weight='normal', labelpad=12)
ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0)
ax4.invert_yaxis() 
draw_atlas_labels(ax4)
fig4.savefig(os.path.join(plot_dir, "matriz_aceptacion_fig7.png"), dpi=300, bbox_inches='tight')

# --- Matriz de pureza ---
sumas_por_columna = matriz_array.sum(axis=0, keepdims=True)
sumas_por_columna[sumas_por_columna == 0] = 1 
matriz_pureza = (matriz_array / sumas_por_columna) * 100

fig5, ax5 = plt.subplots(figsize=(8, 6.5))
divider5 = make_axes_locatable(ax5)
cax5 = divider5.append_axes("right", size="5%", pad=0.15)
sns.heatmap(matriz_pureza, annot=True, fmt=".1f", cmap="Greens",
            xticklabels=[f"Reco {c}" for c in categorias_reco],
            yticklabels=[f"True {c}" for c in categorias_verdad_matriz],
            cbar_ax=cax5, cbar_kws={'label': 'Fraction of reconstructed events (%)'},
            annot_kws={"size": 11, "weight": "normal"},
            square=True, linewidths=1.0, linecolor='white', ax=ax5)
ax5.set_xlabel("Reconstructed category", weight='normal', labelpad=12)
ax5.set_ylabel("True production mode", weight='normal', labelpad=12)
ax5.set_yticklabels(ax5.get_yticklabels(), rotation=0)
ax5.invert_yaxis() 
draw_atlas_labels(ax5)
fig5.savefig(os.path.join(plot_dir, "matriz_pureza_fig8.png"), dpi=300, bbox_inches='tight')


# ============================================================================================
# 6. AJUSTES PARAMÉTRICOS DE SEÑAL
# ============================================================================================
print("\n=================================================================")
print("Iniciando ajustes paramétricos sobre señal MC...")
print("=================================================================")

def add_atlas_fit_label(ax):
    ax.text(0.05, 0.92, "ATLAS", transform=ax.transAxes, fontsize=16, fontweight='bold', fontstyle='italic', ha='left', va='top')
    ax.text(0.17, 0.92, "Simulation", transform=ax.transAxes, fontsize=16, weight='normal', ha='left', va='top')
    ax.text(0.05, 0.85, r"$H \to \gamma\gamma, m_H = 125$ GeV", transform=ax.transAxes, fontsize=14, weight='normal', ha='left', va='top')

def set_professional_axes(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=12)
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)

ruta_txt_mc = os.path.join(plot_dir, "parametros_mc10.txt")
bin_edges = np.arange(100, 161, 1)
bin_centres = np.arange(100.5, 160.5, 1)

with open(ruta_txt_mc, "w", encoding="utf-8") as f_mc:
    f_mc.write("=================================================================\n")
    f_mc.write("             RESULTADOS DE AJUSTES MC (Masa ~125 GeV)            \n")
    f_mc.write("=================================================================\n\n")

    # ----------------------------------------------------------------------
    # 6.A: AJUSTE MODO ggH (Tres Modelos)
    # ----------------------------------------------------------------------
    print("\n>>> AJUSTANDO CATEGORÍA ggH (Gauss, Breit-Wigner, DCB)")
    f_mc.write("--- MODO ggH (Comparación de 3 modelos) ---\n\n")

    mass_ggh = mc_masses["ggH"]
    weights_ggh = mc_weights_fit["ggH"]

    if len(mass_ggh) > 0:
        data_y_ggh, _ = np.histogram(mass_ggh, bins=bin_edges, weights=weights_ggh)
        sumw2_ggh, _ = np.histogram(mass_ggh, bins=bin_edges, weights=weights_ggh**2)
        err_ggh = np.sqrt(sumw2_ggh)

        mask_valido_ggh = data_y_ggh > 0
        x_valido_ggh = bin_centres[mask_valido_ggh]
        y_valido_ggh = data_y_ggh[mask_valido_ggh]
        err_valido_ggh = np.maximum(err_ggh[mask_valido_ggh], 1e-10)

        amplitud_total_ggh = np.sum(y_valido_ggh)

        modelos_ggh = {
            "Gaussian": GaussianModel(prefix='sig_'),
            "Breit-Wigner": BreitWignerModel(prefix='sig_'),
            "Double Crystal Ball": Model(double_cb_pdf, prefix='sig_')
        }

        for nombre_mod, modelo in modelos_ggh.items():
            pars = modelo.make_params()
            pars['sig_center'].set(value=125.0, min=120.0, max=130.0)
            pars['sig_sigma'].set(value=1.5, min=0.5, max=5.0) 
            pars['sig_amplitude'].set(value=max(amplitud_total_ggh, 1e-5), min=0)
            
            if nombre_mod == "Double Crystal Ball":
                pars['sig_beta_L'].set(value=1.5, min=0.1, max=10.0)
                pars['sig_m_L'].set(value=2.0, min=0.1, max=50.0)
                pars['sig_beta_R'].set(value=1.5, min=0.1, max=10.0)
                pars['sig_m_R'].set(value=2.0, min=0.1, max=50.0)

            try:
                out = modelo.fit(y_valido_ggh, pars, x=x_valido_ggh, weights=1/err_valido_ggh, nan_policy='omit')
                print(f"  [ggH - {nombre_mod:18}] Masa: {out.params['sig_center'].value:.2f} GeV | Chi2 Red: {out.redchi:.2f}")
                
                f_mc.write(f"Categoría: ggH | Modelo: {nombre_mod}\n")
                f_mc.write(f"Chi2 Reducido  : {out.redchi:.4f}\n")
                for p_name, p in out.params.items():
                    error = p.stderr if p.stderr is not None else 0.0
                    f_mc.write(f"{p_name:<15}: {p.value:.4f} +/- {error:.4f}\n")
                f_mc.write("\n")

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.errorbar(bin_centres, data_y_ggh, yerr=err_ggh, fmt='ko', label='ggH Data', markersize=5, capsize=0, elinewidth=1.5)
                
                x_smooth = np.linspace(100, 160, 1000) 
                ax.plot(x_smooth, out.eval(x=x_smooth), '-r', linewidth=2.5, label=f'{nombre_mod} Fit')
                
                set_professional_axes(ax)
                ax.set_xlim(100, 150) 
                ax.set_xticks(np.arange(100, 151, 10)) 
                ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]", fontsize=14)
                ax.set_ylabel("Events / 1 GeV", fontsize=14)
                ax.set_title(f"Fit Record: ggH - {nombre_mod}", fontsize=10, color='gray')
                
                add_atlas_fit_label(ax)
                ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.95), frameon=False, fontsize=12)
                ax.set_ylim(bottom=0, top=max(data_y_ggh)*1.2) 
                
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"mc_fit_ggH_{nombre_mod.replace(' ', '')}.png"), dpi=300)
                plt.close()
                
            except Exception as e:
                print(f"  [X] Error ajustando ggH con {nombre_mod}: {e}")

    f_mc.write("-" * 65 + "\n\n")

    # ----------------------------------------------------------------------
    # 6.B: AJUSTE OTRAS CATEGORÍAS (Solo DSCB)
    # ----------------------------------------------------------------------
    f_mc.write("--- AJUSTES OTRAS CATEGORÍAS (Modelo: Double Crystal Ball) ---\n\n")
    modelo_dcb = Model(double_cb_pdf, prefix='sig_')
    categorias_restantes = ["VBF", "VH", "ttH"]

    for cat in categorias_restantes:
        mass_data = mc_masses[cat]
        weights_data = mc_weights_fit[cat]
        
        if len(mass_data) < 10: continue
            
        print(f"\n>>> AJUSTANDO CATEGORÍA: {cat} (Solo DCB)")
        
        data_y, _ = np.histogram(mass_data, bins=bin_edges, weights=weights_data)
        sumw2, _ = np.histogram(mass_data, bins=bin_edges, weights=weights_data**2)
        data_y_errors = np.sqrt(sumw2)
        
        mask_valido = data_y > 0
        x_valido = bin_centres[mask_valido]
        y_valido = data_y[mask_valido]
        err_valido = np.maximum(data_y_errors[mask_valido], 1e-10) 
        
        if len(x_valido) < 5: continue

        amplitud_total = np.sum(y_valido)

        pars = modelo_dcb.make_params()
        pars['sig_center'].set(value=125.0, min=120.0, max=130.0)
        pars['sig_sigma'].set(value=1.5, min=0.5, max=5.0) 
        pars['sig_amplitude'].set(value=max(amplitud_total, 1e-5), min=0)
        pars['sig_beta_L'].set(value=1.5, min=0.1, max=10.0)
        pars['sig_m_L'].set(value=2.0, min=0.1, max=50.0)
        pars['sig_beta_R'].set(value=1.5, min=0.1, max=10.0)
        pars['sig_m_R'].set(value=2.0, min=0.1, max=50.0)
        
        try:
            out = modelo_dcb.fit(y_valido, pars, x=x_valido, weights=1/err_valido, nan_policy='omit')
            print(f"  [{cat:18}] Masa: {out.params['sig_center'].value:.2f} GeV | Chi2 Red: {out.redchi:.2f}")

            f_mc.write(f"Categoría: {cat} | Modelo: Double Crystal Ball\n")
            f_mc.write(f"Chi2 Reducido  : {out.redchi:.4f}\n")
            for p_name, p in out.params.items():
                error = p.stderr if p.stderr is not None else 0.0
                f_mc.write(f"{p_name:<15}: {p.value:.4f} +/- {error:.4f}\n")
            f_mc.write("-" * 65 + "\n\n")

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.errorbar(bin_centres, data_y, yerr=data_y_errors, fmt='ko', label=f'{cat} Data', markersize=5, capsize=0, elinewidth=1.5)
            
            x_smooth = np.linspace(100, 160, 1000)
            ax.plot(x_smooth, out.eval(x=x_smooth), '-r', linewidth=2.5, label='Double Crystal Ball Fit')
            
            set_professional_axes(ax)
            ax.set_xlim(100, 150) 
            ax.set_xticks(np.arange(100, 151, 10))
            ax.set_xlabel(r"$m_{\gamma\gamma}$ [GeV]", fontsize=14)
            ax.set_ylabel("Events / 1 GeV", fontsize=14)
            ax.set_title(f"Fit Record: {cat} - Double Crystal Ball", fontsize=10, color='gray')
            
            add_atlas_fit_label(ax)
            ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.95), frameon=False, fontsize=12)
            ax.set_ylim(bottom=0, top=max(data_y)*1.2)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"mc_fit_{cat}_DoubleCrystalBall.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"  [X] Error ajustando {cat}: {e}")

print(f"\n✓ ¡Ajustes terminados con éxito! Revisa las imágenes y el archivo de parámetros en '{plot_dir}/'")