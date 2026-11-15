import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pybamm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class BatterySimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reprism - PyBaMM Çoklu Karşılaştırma Arayüzü")
        self.root.geometry("1450x880")

        # Her simülasyon için benzersiz renk paleti
        self.color_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        self.color_index = 0

        left_panel = ttk.Frame(root, padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        group1 = ttk.LabelFrame(left_panel, text=" 1. Simülasyon ve Grafik Ayarları ", padding=10)
        group1.pack(fill=tk.X, pady=5)

        ttk.Label(group1, text="İşlem Modu:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.operation_var = tk.StringVar(value="Şarj (Charge)")
        op_dropdown = ttk.Combobox(
            group1,
            textvariable=self.operation_var,
            values=["Deşarj (Discharge)", "Şarj (Charge)"],
            state="readonly",
            width=28
        )
        op_dropdown.grid(row=0, column=1, pady=2)
        op_dropdown.bind("<<ComboboxSelected>>", self.on_operation_change)

        ttk.Label(group1, text="Y-Ekseni Değişkeni:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.y_axis_var = tk.StringVar(value="Anot Yüzey Potansiyeli [V]")
        y_dropdown = ttk.Combobox(
            group1,
            textvariable=self.y_axis_var,
            values=["Hücre Gerilimi [V]", "Anot Yüzey Potansiyeli [V]"],
            state="readonly",
            width=28
        )
        y_dropdown.grid(row=1, column=1, pady=2)
        y_dropdown.bind("<<ComboboxSelected>>", lambda e: self.reset_plot_axes())

        group2 = ttk.LabelFrame(left_panel, text=" 2. Çalışma Koşulları ", padding=10)
        group2.pack(fill=tk.X, pady=5)

        ttk.Label(group2, text="Ortam Sıcaklığı (°C):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.temp_var = tk.StringVar(value="25")
        ttk.Entry(group2, textvariable=self.temp_var, width=12).grid(row=0, column=1, pady=2, sticky=tk.W)

        ttk.Label(group2, text="Akım Oranı (C-rate):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.crate_var = tk.StringVar(value="1.0")
        ttk.Entry(group2, textvariable=self.crate_var, width=12).grid(row=1, column=1, pady=2, sticky=tk.W)

        ttk.Label(group2, text="Başlangıç SoC (%):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.soc_var = tk.StringVar(value="10")
        ttk.Entry(group2, textvariable=self.soc_var, width=12).grid(row=2, column=1, pady=2, sticky=tk.W)

        ttk.Label(group2, text="Üst Gerilim Sınırı (V):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.upper_v_var = tk.StringVar(value="4.2")
        ttk.Entry(group2, textvariable=self.upper_v_var, width=12).grid(row=3, column=1, pady=2, sticky=tk.W)

        ttk.Label(group2, text="Alt Gerilim Sınırı (V):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.lower_v_var = tk.StringVar(value="2.5")
        ttk.Entry(group2, textvariable=self.lower_v_var, width=12).grid(row=4, column=1, pady=2, sticky=tk.W)

        ttk.Label(group2, text="Hazır Parametre Seti:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.param_set_var = tk.StringVar(value="Chen2020")
        param_dropdown = ttk.Combobox(
            group2,
            textvariable=self.param_set_var,
            values=["Chen2020", "OKane2022", "Prada2013", "Marquis2019"],
            state="readonly",
            width=12
        )
        param_dropdown.grid(row=5, column=1, pady=2, sticky=tk.W)
        param_dropdown.bind("<<ComboboxSelected>>", self.on_param_set_change)

        self.thermal_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(group2, text="Termal Modeli Aktif Et", variable=self.thermal_var).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=4)

        group3 = ttk.LabelFrame(left_panel, text=" 3. Yapısal Parametreler ", padding=10)
        group3.pack(fill=tk.X, pady=5)

        self.param_entries = {}
        fields = [
            ("Elektrolit Derişimi [mol/m³]", "Initial concentration in electrolyte [mol.m-3]", "1000"),
            ("Anot Porozitesi (-)", "Negative electrode porosity", "0.25"),
            ("Katot Porozitesi (-)", "Positive electrode porosity", "0.335"),
            ("Ayırıcı Porozitesi (-)", "Separator porosity", "0.47"),
            ("Anot Kalınlığı [m]", "Negative electrode thickness [m]", "8.52e-05"),
            ("Katot Kalınlığı [m]", "Positive electrode thickness [m]", "7.56e-05"),
            ("Ayırıcı Kalınlığı [m]", "Separator thickness [m]", "1.2e-05"),
        ]

        for idx, (label_text, key, default_val) in enumerate(fields):
            ttk.Label(group3, text=label_text + ":").grid(row=idx, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar(value=default_val)
            ttk.Entry(group3, textvariable=var, width=12).grid(row=idx, column=1, pady=2)
            self.param_entries[key] = var

    
        group4 = ttk.Frame(left_panel, padding=5)
        group4.pack(fill=tk.X, pady=5)

        self.overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(group4, text="Grafikleri Üst Üste Ekle (Karşılaştır)", variable=self.overlay_var).pack(anchor=tk.W, pady=3)

        ttk.Button(group4, text="▶ Simülasyonu Çalıştır", command=self.run_simulation).pack(fill=tk.X, pady=3)
        ttk.Button(group4, text="🗑 Grafiği Temizle", command=self.clear_plot).pack(fill=tk.X, pady=2)

        self.status_label = ttk.Label(group4, text="Hazır", font=("Arial", 9, "italic"), foreground="gray")
        self.status_label.pack(fill=tk.X, pady=5)

        
        plot_frame = ttk.Frame(root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        self.reset_plot_axes()

    def on_param_set_change(self, event=None):
        """Seçilen parametre setine göre voltaj sınırlarını otomatik ayarlar."""
        selected_set = self.param_set_var.get()
        if selected_set == "Prada2013":
            self.upper_v_var.set("3.6")
            self.lower_v_var.set("2.0")
            self.thermal_var.set(False)
            self.status_label.config(text="Prada2013 (LFP): Voltaj 3.6V/2.0V ve Termal Kapatıldı.", foreground="orange")
        else:
            self.upper_v_var.set("4.2")
            self.lower_v_var.set("2.5")
            self.thermal_var.set(True)
            self.status_label.config(text=f"{selected_set} seçildi.", foreground="gray")

    def on_operation_change(self, event=None):
        if "Deşarj" in self.operation_var.get():
            self.soc_var.set("100")
            self.y_axis_var.set("Hücre Gerilimi [V]")
        else:
            self.soc_var.set("10")
            self.y_axis_var.set("Anot Yüzey Potansiyeli [V]")
        self.reset_plot_axes()

    def get_float(self, var, name):
        try:
            return float(var.get())
        except ValueError:
            raise ValueError(f"'{name}' alanına geçerli bir sayı giriniz.")

    def reset_plot_axes(self):
        self.ax.clear()
        self.color_index = 0
        is_discharge = "Deşarj" in self.operation_var.get()
        y_selection = self.y_axis_var.get()

        x_label = "Deşarj Kapasitesi [Ah]" if is_discharge else "Şarj Kapasitesi [Ah]"
        self.ax.set_xlabel(x_label, fontsize=11)
        self.ax.set_ylabel(y_selection, fontsize=11)

        if y_selection == "Anot Yüzey Potansiyeli [V]":
            self.ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="Lityum Kaplama Eşiği (0 V)")
            title = f"Anot Yüzey Potansiyeli Analizi ({'Deşarj' if is_discharge else 'Şarj'})"
        else:
            title = f"Hücre Gerilimi Karşılaştırması ({'Deşarj' if is_discharge else 'Şarj'})"

        self.ax.set_title(title, fontsize=13)
        self.ax.grid(True, linestyle=":", alpha=0.7)
        self.ax.legend(loc="upper right")
        self.canvas.draw()

    def clear_plot(self):
        self.reset_plot_axes()
        self.status_label.config(text="Grafik temizlendi.", foreground="gray")

    def get_anode_potential(self, sol, model):
        all_vars = list(model.variables.keys())
        candidates = [
            "Negative electrode surface potential difference [V]",
            "Negative electrode surface potential difference at separator interface [V]",
            "Negative electrode potential [V]",
            "Negative electrode surface potential [V]"
        ]
        for cand in candidates:
            if cand in all_vars:
                return sol[cand].entries

        for v in all_vars:
            if "negative electrode" in v.lower() and "potential" in v.lower():
                return sol[v].entries

        raise KeyError("Anot yüzey potansiyeli değişkeni bulunamadı.")

    def get_capacity(self, sol, model, is_discharge):
        all_vars = list(model.variables.keys())
        candidates = ["Discharge capacity [A.h]"] if is_discharge else ["Charge capacity [A.h]"]
        candidates.extend(["Capacity [A.h]", "Throughput capacity [A.h]"])

        for cand in candidates:
            if cand in all_vars:
                return sol[cand].entries

        for v in all_vars:
            if "capacity [a.h]" in v.lower():
                return sol[v].entries

        t_hours = sol["Time [h]"].entries
        current = np.abs(sol["Current [A]"].entries)
        return np.trapz(current, t_hours) if len(t_hours) > 1 else t_hours

    def run_simulation(self):
        try:
            T_celsius = self.get_float(self.temp_var, "Ortam Sıcaklığı")
            T_kelvin = T_celsius + 273.15
            C_rate = self.get_float(self.crate_var, "C-rate")
            soc_val = max(0.001, min(0.999, self.get_float(self.soc_var, "Başlangıç SoC") / 100.0))
            upper_v = self.get_float(self.upper_v_var, "Üst Gerilim Sınırı")
            lower_v = self.get_float(self.lower_v_var, "Alt Gerilim Sınırı")
            param_set_name = self.param_set_var.get()

            is_discharge = "Deşarj" in self.operation_var.get()
            y_selection = self.y_axis_var.get()

            self.status_label.config(text=f"{param_set_name} | {T_celsius}°C hesaplanıyor...", foreground="blue")
            self.root.update()

            
            if param_set_name == "Prada2013":
                options = {"thermal": "isothermal"}
            else:
                options = {"thermal": "lumped"} if self.thermal_var.get() else {"thermal": "isothermal"}
            
            model = pybamm.lithium_ion.DFN(options=options)
            param = pybamm.ParameterValues(param_set_name)

            update_dict = {
                "Upper voltage cut-off [V]": upper_v,
                "Lower voltage cut-off [V]": lower_v,
            }

          
            if "Ambient temperature [K]" in param:
                update_dict["Ambient temperature [K]"] = T_kelvin
            if "Initial temperature [K]" in param:
                update_dict["Initial temperature [K]"] = T_kelvin

           
            for pybamm_key, var in self.param_entries.items():
                if pybamm_key in param:
                    try:
                        update_dict[pybamm_key] = self.get_float(var, pybamm_key)
                    except ValueError:
                        pass

            param.update(update_dict, check_already_exists=False)

            if is_discharge:
                experiment = pybamm.Experiment([f"Discharge at {C_rate} C until {lower_v} V"])
            else:
                experiment = pybamm.Experiment([f"Charge at {C_rate} C until {upper_v} V"])

            sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
            sol = sim.solve(initial_soc=soc_val)

            capacity = self.get_capacity(sol, model, is_discharge)

            if y_selection == "Hücre Gerilimi [V]":
                y_data = sol["Terminal voltage [V]"].entries
            else:
                y_data = self.get_anode_potential(sol, model)

            if y_data.ndim > 1:
                y_data = np.min(y_data, axis=0)

            if not self.overlay_var.get():
                self.reset_plot_axes()

            op_tag = "Deşarj" if is_discharge else "Şarj"
            current_color = self.color_palette[self.color_index % len(self.color_palette)]
            self.color_index += 1

            label_text = f"{param_set_name} | {T_celsius:.0f}°C | {C_rate}C {op_tag}"

            self.ax.plot(capacity, y_data, label=label_text, color=current_color, linewidth=2)
            self.ax.legend(loc="upper right")
            self.canvas.draw()

            self.status_label.config(text=f"✅ {param_set_name} Başarıyla Çalıştırıldı", foreground="green")

        except Exception as e:
            messagebox.showerror("Simülasyon Hatası", f"Simülasyon çalıştırılamadı:\n\n{str(e)}")
            self.status_label.config(text="Hata oluştu.", foreground="red")


if __name__ == "__main__":
    root = tk.Tk()
    app = BatterySimApp(root)
    root.mainloop()
