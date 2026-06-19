# ============================================================================================
# ======== Instalación de paquetes en el ordenador (sólo es necesario la primera vez) ========
# import sys
# %pip install atlasopenmagic
# from atlasopenmagic import install_from_environment
# install_from_environment()
# ============================================================================================

# ============================================================================================
# 1. LIBRERÍAS Y CONFIGURACIÓN INICIAL
# ============================================================================================
import uproot 
import awkward as ak 
import numpy as np 
import matplotlib.pyplot as plt 
import vector 
import atlasopenmagic as atom 
import os 
from lmfit.models import PolynomialModel
from lmfit import Model 

# --- Configuración de ATLAS Open Data, acceso a los datos ---
atom.set_release('2025e-13tev-beta')
skim = "GamGam" 
files_list = atom.get_urls('data', skim, protocol='https', cache=True) 

variables = [
    "photon_pt", "photon_eta", "photon_phi", "photon_e", "photon_isTightID", "photon_ptcone20",
    "jet_pt", "jet_eta", "jet_phi", "jet_e", "jet_btag_quantile", 
    "lep_n", "lep_pt", "lep_eta", "lep_phi", "lep_e", 
    "met", "met_phi"
]                        

# --- Configuración de variables para los histogramas de análisis ---
fraction = 1.0  
xmin = 100 
xmax = 160 
step_size = 1 
bin_edges = np.arange(start=xmin, stop=xmax+step_size, step=step_size)
bin_centres = np.arange(start=xmin+step_size/2, stop=xmax+step_size/2, step=step_size)

# --- Parámetros de ajuste y amplitudes (obtenidos del MC) ---
mc_params = {
    "ggH":       {"sig": 1.7711, "bL": 1.3602, "mL": 10.9674, "bR": 1.4462, "mR": 50.0000},
    "VBF":       {"sig": 1.5203, "bL": 1.3705, "mL": 7.0314,  "bR": 1.2736, "mR": 50.0000},
    "VH":        {"sig": 1.4990, "bL": 1.4772, "mL": 5.6600,  "bR": 1.3714, "mR": 50.0000},
    "ttH":       {"sig": 1.3593, "bL": 1.6424, "mL": 4.2543,  "bR": 1.4608, "mR": 50.0000},
    "Inclusive": {"sig": 1.7711, "bL": 1.3602, "mL": 10.9674, "bR": 1.4462, "mR": 50.0000} 
}

fracciones_amplitud = {
    "Inclusive": 0.005, 
    "ttH": 0.05,        
    "VH": 0.015,        
    "VBF": 0.02,        
    "ggH": 0.004        
}

plot_dir = "plots_12"
os.makedirs(plot_dir, exist_ok=True)


# ============================================================================================
# 2. FUNCIONES PARA LA SELECCIÓN DE EVENTOS
# ============================================================================================

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


# ============================================================================================
# 3. FUNCIONES PARA LA CATEGORIZACIÓN
# ============================================================================================

def get_clean_jets_mask(jet_pt):
    return (jet_pt > 25) 

def count_bjets(jet_btag_quantile, clean_mask):
    is_bjet = (jet_btag_quantile >= 3) & clean_mask 
    return ak.sum(is_bjet, axis=1)

def calc_mjj(jet_pt, jet_eta, jet_phi, jet_e):
    p4 = vector.zip({"pt": jet_pt, "eta": jet_eta, "phi": jet_phi, "e": jet_e})
    p4_padded = ak.pad_none(p4, 2, axis=1)
    dijet = p4_padded[:, 0] + p4_padded[:, 1]
    return ak.fill_none(dijet.M, 0.0)

def calc_mll(lep_pt, lep_eta, lep_phi, lep_e):
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "e": lep_e})
    p4_padded = ak.pad_none(p4, 2, axis=1) 
    dilep = p4_padded[:, 0] + p4_padded[:, 1]
    return ak.fill_none(dilep.M, 0.0)

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
    return (tth_lep | tth_had), tth_lep, tth_had

def is_vh(lep_n, n_jets, mjj, m_ll, m_lgamma1, m_lgamma2):
    vh_dilep = (lep_n >= 2) & (m_ll >= 70.0) & (m_ll <= 110.0)
    pass_veto_1 = np.abs(m_lgamma1 - 89.0) > 5.0 
    pass_veto_2 = np.abs(m_lgamma2 - 89.0) > 5.0
    vh_lep = (lep_n == 1) & pass_veto_1 & pass_veto_2
    vh_had = (lep_n == 0) & (n_jets >= 2) & (mjj > 60.0) & (mjj < 120.0)
    return (vh_dilep | vh_lep | vh_had), vh_dilep, vh_lep, vh_had

def is_vbf(n_jets, jet_eta, eta_gg, mjj):
    mask_2j = (n_jets >= 2) 
    eta_padded = ak.pad_none(jet_eta, 2, axis=1) 
    eta_j1 = eta_padded[:, 0]
    eta_j2 = eta_padded[:, 1]
    delta_eta = ak.fill_none(np.abs(eta_j1 - eta_j2) > 2, False) 
    zeppenfeld = ak.fill_none(calc_zeppenfeld(eta_gg, eta_j1, eta_j2) < 5, False)
    mass_cut = (mjj > 250.0)
    return mask_2j & delta_eta & zeppenfeld & mass_cut


# ============================================================================================
# 4. SELECCIÓN DE EVENTOS
# ============================================================================================

cat_masses = {"Inclusive": [], "ttH": [], "VH": [], "VBF": [], "ggH": []}

cut_flow = {
    "Total": 0, "Calidad": 0, "pT_Fisico": 0, "Aislamiento": 0, 
    "Transicion_Eta": 0, "Ventana_Masa": 0, "Masa_Iso_Final": 0,
    "-> Categoria: ttH": 0, "   ttH: Leptonico": 0, "   ttH: Hadronico": 0,
    "-> Categoria: VH": 0, "   VH: Dilepton": 0, "   VH: 1 Lepton": 0, "   VH: Hadronico": 0,
    "-> Categoria: VBF": 0, "-> Categoria: ggH": 0
}

for afile in files_list:
    nombre_archivo = afile.split("/")[-1]
    print(f'Intentando procesar: {nombre_archivo}')
    
    try:
        tree = uproot.open(afile + ":analysis") 
        for data in tree.iterate(variables, library="ak", entry_stop=tree.num_entries*fraction):
            cut_flow["Total"] += len(data)
            
            # --- Cortes de fotones ---
            data = data[cut_photon_reconstruction(data['photon_isTightID'])]
            cut_flow["Calidad"] += len(data)

            data = data[cut_photon_pt(data['photon_pt'])]
            cut_flow["pT_Fisico"] += len(data)

            data = data[cut_isolation_pt(data['photon_ptcone20'], data['photon_pt'])]
            cut_flow["Aislamiento"] += len(data)
            
            data = data[cut_photon_eta_transition(data['photon_eta'])]
            cut_flow["Transicion_Eta"] += len(data)
            
            data['mass'] = calc_mass(data['photon_pt'], data['photon_eta'], data['photon_phi'], data['photon_e'])
            data = data[(data['mass'] > xmin) & (data['mass'] < xmax)]
            cut_flow["Ventana_Masa"] += len(data)
            
            data = data[cut_iso_mass(data['photon_pt'], data['mass'])]
            cut_flow["Masa_Iso_Final"] += len(data)

            # --- Cinemática para categorización ---
            clean_mask = get_clean_jets_mask(data['jet_pt'])
            n_jets = ak.sum(clean_mask, axis=1)
            n_bjets = count_bjets(data['jet_btag_quantile'], clean_mask)
            
            mjj = calc_mjj(data['jet_pt'], data['jet_eta'], data['jet_phi'], data['jet_e'])
            eta_gg = calc_eta_gg(data['photon_pt'], data['photon_eta'], data['photon_phi'], data['photon_e'])
            m_ll = calc_mll(data['lep_pt'], data['lep_eta'], data['lep_phi'], data['lep_e'])
            m_lgamma1, m_lgamma2 = calc_mlgamma_masses(data['lep_pt'], data['lep_eta'], data['lep_phi'], data['lep_e'],
                                                       data['photon_pt'], data['photon_eta'], data['photon_phi'], data['photon_e'])

            # --- Aplicación de categorías (por orden de sección eficaz) ---
            m_tth, tth_lep, tth_had = is_tth(data['lep_n'], n_jets, n_bjets)
            
            base_vh = ~m_tth
            m_vh_raw, vh_dilep, vh_lep, vh_had = is_vh(data['lep_n'], n_jets, mjj, m_ll, m_lgamma1, m_lgamma2)
            m_vh = m_vh_raw & base_vh
            
            m_vbf = is_vbf(n_jets, data['jet_eta'], eta_gg, mjj) & ~m_tth & ~m_vh
            m_ggh = (~m_tth) & (~m_vh) & (~m_vbf) 

            # --- Registro  ---
            cut_flow["-> Categoria: ttH"] += ak.sum(m_tth)
            cut_flow["   ttH: Leptonico"] += ak.sum(tth_lep)
            cut_flow["   ttH: Hadronico"] += ak.sum(tth_had)

            cut_flow["-> Categoria: VH"] += ak.sum(m_vh)
            cut_flow["   VH: Dilepton"] += ak.sum(vh_dilep & base_vh)
            cut_flow["   VH: 1 Lepton"] += ak.sum(vh_lep & base_vh)
            cut_flow["   VH: Hadronico"] += ak.sum(vh_had & base_vh)

            cut_flow["-> Categoria: VBF"] += ak.sum(m_vbf)
            cut_flow["-> Categoria: ggH"] += ak.sum(m_ggh)

            # --- Almacenamiento de masas ---
            cat_masses["Inclusive"].append(data['mass'])
            cat_masses["ttH"].append(data['mass'][m_tth])
            cat_masses["VH"].append(data['mass'][m_vh])
            cat_masses["VBF"].append(data['mass'][m_vbf])
            cat_masses["ggH"].append(data['mass'][m_ggh])
            
        print(f"✓ {nombre_archivo} procesado correctamente.")

    except Exception as e:
        print(f"⚠ Saltando {nombre_archivo} por error de red: {e}")
        continue 

# Concatenar arrays de masa
for c in cat_masses:
    cat_masses[c] = ak.concatenate(cat_masses[c]) if len(cat_masses[c]) > 0 else ak.Array([]) 


# ============================================================================================
# 5. AJUSTES Y PLOTS DE CADA CATEGORIA
# ============================================================================================

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

print(f"\nLos plots y resultados se guardarán en la carpeta: '{plot_dir}/'")
ruta_txt = os.path.join(plot_dir, "plots_12.txt")

with open(ruta_txt, "w", encoding="utf-8") as f_txt:
    f_txt.write("=================================================================\n")
    f_txt.write("      RESUMEN DE AJUSTES (DSCB + Poly) - DATOS ATLAS        \n")
    f_txt.write("=================================================================\n\n")

    for cat, all_data in cat_masses.items():
        if len(all_data) < 20: 
            continue
        
        print(f"\n>>> AJUSTANDO CATEGORÍA: {cat} ({len(all_data)} eventos)")
        
        data_x, _ = np.histogram(ak.to_numpy(all_data), bins=bin_edges)
        data_x_errors = np.sqrt(data_x)
        data_x_errors[data_x_errors == 0] = 1.0 

        # --- Definición del modelo (Double CB + Polinomio) ---
        sig_model = Model(double_cb_pdf, prefix='sig_')
        order = 4 if cat in ["Inclusive", "ggH"] else 2
        bkg_model = PolynomialModel(order, prefix='bkg_')
        
        model = bkg_model + sig_model
        pars = model.make_params()
        
        # --- Configuración de parámetros ---
        p = mc_params.get(cat, mc_params["ggH"])

        # Masa y resolución libres pero acotadas
        pars['sig_center'].set(value=125.09, min=120.0, max=130.0, vary=True)
        pars['sig_sigma'].set(value=p["sig"], min=1.0, max=3.0, vary=True)

        # Masa y resolucion. Masa libre solo en Inclusive, ggH
        #if cat in ["Inclusive", "ggH"]:
        #    pars['sig_center'].set(value=125.09, min=120.0, max=130.0, vary=True)
        #else:
        #    pars['sig_center'].set(value=125.09, vary=False)

        #pars['sig_sigma'].set(value=p["sig"], min=1.0, max=3.0, vary=True)

        # Colas libres para categorías con más eventos
        #if cat in ["Inclusive", "ggH"]:
        #    pars['sig_beta_L'].set(value=p["bL"], min=0.5, max=5.0, vary=True)
        #    pars['sig_m_L'].set(value=p["mL"], min=1.0, max=30.0, vary=True)
        #    pars['sig_beta_R'].set(value=p["bR"], min=0.5, max=5.0, vary=True)
        #    pars['sig_m_R'].set(value=p["mR"], min=1.0, max=100.0, vary=True)
        #else:
            # Fijadas las colas para baja estadística (ttH, VH, VBF)
        #    pars['sig_beta_L'].set(value=p["bL"], vary=False)
        #    pars['sig_m_L'].set(value=p["mL"], vary=False)
        #    pars['sig_beta_R'].set(value=p["bR"], vary=False)
        #    pars['sig_m_R'].set(value=p["mR"], vary=False)

        # Colas fijadas para todas las categorías 
        pars['sig_beta_L'].set(value=p["bL"], vary=False)
        pars['sig_m_L'].set(value=p["mL"], vary=False)
        pars['sig_beta_R'].set(value=p["bR"], vary=False)
        pars['sig_m_R'].set(value=p["mR"], vary=False)  

        # Amplitud
        amp_inicial = len(all_data) * fracciones_amplitud.get(cat, 0.005)
        pars['sig_amplitude'].set(value=amp_inicial, min=0, vary=True)
        
        # Fondo libre pero con valor inicial en la media de los datos
        pars['bkg_c0'].set(value=np.mean(data_x))
        for i in range(1, order + 1): 
            pars[f'bkg_c{i}'].set(value=0)

        # --- Ejecución del Ajuste ---
        try:
            out = model.fit(data_x, pars, x=bin_centres, weights=1/data_x_errors, nan_policy='omit')
            if not out.success:
                print(f"Advertencia: El ajuste para {cat} no convergió del todo.")
        except Exception as e:
            print(f"Error crítico ajustando {cat}: {e}")
            continue

        # --- Extracción de resultados ---
        res = out.params.valuesdict()
        S = res.get('sig_amplitude', 0)
        S_err = out.params['sig_amplitude'].stderr if out.params['sig_amplitude'].stderr else 0.0
        mH_err = out.params['sig_center'].stderr if out.params['sig_center'].stderr else 0.0
        res_err = out.params['sig_sigma'].stderr if out.params['sig_sigma'].stderr else 0.0

        mask_window = (bin_centres >= 120) & (bin_centres <= 130)
        comps_bins = out.eval_components(x=bin_centres)
        
        sig_fit_window = comps_bins['sig_'][mask_window]
        bkg_fit_window = comps_bins['bkg_'][mask_window]
        bkg_valido = np.maximum(bkg_fit_window, 1.0) # Evitar división por cero
        
        # Significancia s_i = S_i / sqrt(B_i)
        s_i = sig_fit_window / np.sqrt(bkg_valido)
        significancia = np.sqrt(np.sum(s_i**2))

        # --- Escritura en archivo ---
        f_txt.write(f"Categoría: {cat} | Señal: DoubleCrystalBall | Fondo: Polynomial (Grado {order})\n")
        f_txt.write(f"  Masa H            : {res.get('sig_center', 0):.4f} +/- {mH_err:.4f} GeV\n")      
        f_txt.write(f"  Resolución        : {res.get('sig_sigma', 0):.4f} +/- {res_err:.4f} GeV\n")    
        f_txt.write(f"  Eventos Señal (S) : {S:.2f} +/- {S_err:.2f}\n")
        f_txt.write(f"  Significancia     : {significancia:.4f} sigma\n")
        f_txt.write(f"  Chi2 Reducido     : {out.redchi:.4f}\n")
        f_txt.write("  Parámetros Libres Ajustados:\n")
        for name, param in out.params.items():
            if param.vary: 
                err = param.stderr if param.stderr else 0.0
                f_txt.write(f"    - {name}: {param.value:.5g} +/- {err:.5g}\n")
        f_txt.write("-" * 65 + "\n\n")

        # --- Plotting ---
        plt.figure(figsize=(8, 8))
        
        # Plot principal
        main_ax = plt.axes([0.15, 0.35, 0.8, 0.6])
        main_ax.errorbar(bin_centres, data_x, yerr=data_x_errors, fmt='ko', label=f'{cat} Data', markersize=4)
        
        x_dense = np.linspace(xmin, xmax, 1000) 
        fit_dense = model.eval(out.params, x=x_dense)
        bkg_dense = model.eval_components(params=out.params, x=x_dense)['bkg_']
        
        main_ax.plot(x_dense, fit_dense, '-r', label='Signal + Background')
        main_ax.plot(x_dense, bkg_dense, '--r', label='Background')
        
        main_ax.set_title(f"Channel: {cat} | Sig: DoubleCB | Bkg: Poly{order}")
        main_ax.set_ylabel(f'Events / {step_size} GeV')
        main_ax.legend(frameon=False)
        
        # Textos en el gráfico
        atlas_text = (
            r"$\mathbfit{ATLAS}$ Open Data" + "\n" +
            r"$\sqrt{s}=13$ TeV, $36.1$ fb$^{-1}$" + "\n" +
            r"$H \rightarrow \gamma\gamma, m_H = 125.09$ GeV"
        )
        main_ax.text(0.05, 0.20, atlas_text, transform=main_ax.transAxes, fontsize=12, 
                     verticalalignment='top', bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))

        
        # Plot de residuos
        sub_ax = plt.axes([0.15, 0.1, 0.8, 0.2])
        bkg_fit_points = out.eval_components(x=bin_centres)['bkg_']
        sub_ax.errorbar(bin_centres, data_x - bkg_fit_points, yerr=data_x_errors, fmt='ko', markersize=4)
        
        sig_dense = fit_dense - bkg_dense
        sub_ax.plot(x_dense, sig_dense, '-r') 
        sub_ax.axhline(0, color='red', linestyle='--')
        
        sub_ax.set_xlabel(r'$m_{\gamma\gamma}$ [GeV]')
        sub_ax.set_ylabel('Events - Bkg')
        
        # Guardar gráfico
        plt.savefig(os.path.join(plot_dir, f"{cat}_FitReal.png"), dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================================
# 6. GUARDADO DE TABLA DE CORTES Y EVENTOS EN CADA CATEGORÍA
# ============================================================================================

ruta_tabla = os.path.join(plot_dir, "tabla_eventos.txt")

with open(ruta_tabla, "w", encoding="utf-8") as f_tabla:
    # ---------------------------------------------------------
    # TABLA 1: SELECCION DE EVENTOS
    # ---------------------------------------------------------
    f_tabla.write("SELECCIÓN DE EVENTOS\n")
    ancho_total = 164
    header_1 = "=" * ancho_total
    header_2 = f"{'CORTE':<32} | {'Nº DE EVENTOS RESTANTES':<23} | {'Nº DE EVENTOS DESCARTADOS CON EL CORTE':<38} | {'% DE EVENTOS DESCARTADOS CON EL CORTE':<37} | {'% DE EVENTOS RESTANTES':<22}"
    header_3 = "-" * ancho_total

    f_tabla.write(header_1 + "\n")
    f_tabla.write(header_2 + "\n")
    f_tabla.write(header_3 + "\n")

    total_eventos = cut_flow["Total"]
    eventos_previos = total_eventos

    pasos_base = ["Total", "Calidad", "pT_Fisico", "Aislamiento", "Transicion_Eta", "Ventana_Masa", "Masa_Iso_Final"]

    for paso in pasos_base:
        cuenta = cut_flow[paso]
        if total_eventos == 0: continue

        extraidos = eventos_previos - cuenta if paso != "Total" else 0
        pct_previo = (cuenta / eventos_previos) * 100 if eventos_previos > 0 else 0
        pct_total = (cuenta / total_eventos) * 100
        eventos_previos = cuenta
        
        str_extraidos = str(extraidos) if paso != "Total" else "-"
        str_pct_previo = f"{pct_previo:>35.2f} %" if paso != "Total" else "100.00 %".rjust(37)

        linea = f"{paso:<32} | {cuenta:<23} | {str_extraidos:<38} | {str_pct_previo} | {pct_total:>20.2f} %"
        f_tabla.write(linea + "\n")

    f_tabla.write("=" * ancho_total + "\n\n\n")

    # ---------------------------------------------------------
    # TABLA 2: DISTRIBUCIÓN EN CATEGORÍAS
    # ---------------------------------------------------------
    f_tabla.write("DISTRIBUCIÓN EN CATEGORÍAS \n")
    ancho_cat = 65
    f_tabla.write("=" * ancho_cat + "\n")
    f_tabla.write(f"{'CATEGORÍA / SUBCATEGORÍA':<32} | {'EVENTOS':<12} | {'% DEL FINAL':<15}\n")
    f_tabla.write("-" * ancho_cat + "\n")

    eventos_finales = cut_flow["Masa_Iso_Final"]

    for paso, cuenta in cut_flow.items():
        if "Categoria" in paso or "   " in paso:
            pct_final = (cuenta / eventos_finales) * 100 if eventos_finales > 0 else 0
            
            if "   " in paso:
                f_tabla.write(f"{paso:<32} | {cuenta:<12} | {pct_final:>12.2f} %\n")
            else:
                f_tabla.write(f"{paso:<32} | {cuenta:<12} | {pct_final:>12.2f} %\n")

    f_tabla.write("=" * ancho_cat + "\n")

print(f"\n✓ El resumen de cortes y categorías se ha guardado en: '{ruta_tabla}'")