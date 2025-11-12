import os
import io
import sys
import math
import datetime as dt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from statsmodels.tsa.stattools import adfuller
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.formula.api import ols
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm

# ---- Styling ----
sns.set_theme(style="whitegrid", font_scale=1.1)
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

APP_TITLE = "Walmart Sales Analysis Dashboard"
APP_SIZE = "1280x860"

def nice_number(n):
    try:
        return f"{n:,.3f}" if isinstance(n, (int, float)) else str(n)
    except Exception:
        return str(n)

def safe_pct_change(s):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = s.pct_change()
    return r.replace([np.inf, -np.inf], np.nan)

def model_summary_as_text(model):
    buf = io.StringIO()
    buf.write(model.summary().as_text())
    return buf.getvalue()

class WalmartApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(APP_SIZE)
        self.df = None
        self.results_df = None
        self.model = None
        self.figures = []
        self._build_ui()

    def _build_ui(self):
        # Menu
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open CSV", command=self.load_data, accelerator="Ctrl+O")
        filemenu.add_command(label="Export Results", command=self.export_results, state="disabled")
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)
        self.filemenu = filemenu

        # Toolbar
        toolbar = ttk.Frame(self.root, padding=(10, 8))
        ttk.Button(toolbar, text="📂 Load Data", command=self.load_data).pack(side="left", padx=(0, 10))
        self.path_label_var = tk.StringVar(value="No file selected")
        ttk.Label(toolbar, textvariable=self.path_label_var, font=("Segoe UI", 10)).pack(side="left")
        toolbar.pack(fill="x")

        # Notebook
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=10, pady=5)

        self.tab_summary = ttk.Frame(self.nb)
        self.tab_charts = ttk.Frame(self.nb)
        self.tab_acf = ttk.Frame(self.nb)
        self.tab_reg = ttk.Frame(self.nb)
        self.tab_extra = ttk.Frame(self.nb)

        self.nb.add(self.tab_summary, text="📊 Efficiency Summary")
        self.nb.add(self.tab_charts, text="📈 Visualizations")
        self.nb.add(self.tab_acf, text="🔁 Autocorrelation")
        self.nb.add(self.tab_reg, text="📘 Regression Analysis")
        self.nb.add(self.tab_extra, text="🎨 Additional Insights")

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=(5, 2))
        status.pack(side="bottom", fill="x")

        self.root.bind_all("<Control-o>", lambda e: self.load_data())

    def process_dataset(self, df):
        if 'Date' not in df.columns:
            raise ValueError("Missing 'Date' column.")
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date']).copy()

        store_results = []
        for store_id in sorted(df['Store'].unique()):
            store_df = df[df['Store'] == store_id].sort_values('Date')
            store_df['Sales_Returns'] = safe_pct_change(store_df['Weekly_Sales'])
            returns = store_df['Sales_Returns'].dropna()

            if len(returns) > 10:
                adf_stat, adf_p, *_ = adfuller(returns, maxlag=10)
                run_z, run_p = runstest_1samp(returns, cutoff='mean')
                lb = acorr_ljungbox(returns, lags=[10], return_df=True)
                lb_p = float(lb['lb_pvalue'].iloc[0]) if not lb.empty else np.nan

                store_results.append({
                    'Store': store_id,
                    'ADF p-value': nice_number(adf_p),
                    'ADF Result': 'Efficient' if pd.notna(adf_p) and adf_p > 0.05 else 'Inefficient',
                    'Runs p-value': nice_number(run_p),
                    'Runs Result': 'Efficient' if pd.notna(run_p) and run_p > 0.05 else 'Inefficient',
                    'Ljung-Box p-value': nice_number(lb_p),
                    'LB Result': 'No Autocorr' if pd.notna(lb_p) and lb_p > 0.05 else 'Autocorr'
                })

        results_df = pd.DataFrame(store_results)
        fml = 'Weekly_Sales ~ Holiday_Flag + Temperature + Fuel_Price + CPI + Unemployment + Holiday_Flag:CPI'
        model = ols(fml, data=df).fit()

        return df, results_df, model

    def load_data(self):
        csv_path = filedialog.askopenfilename(title="Select Walmart CSV", filetypes=[("CSV Files", "*.csv")])
        if not csv_path:
            return
        self._set_status("Loading data...")
        self.root.update_idletasks()
        try:
            df = pd.read_csv(csv_path)
            required = {'Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
            self.df, self.results_df, self.model = self.process_dataset(df)
            self.path_label_var.set(os.path.basename(csv_path))
            self._build_summary_tab()
            self._build_charts_tab()
            self._build_acf_tab()
            self._build_reg_tab()
            self._build_extra_tab()
            self.filemenu.entryconfig("Export Results", state="normal")
            self._set_status("Data loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self._set_status("Error loading data")

    def _clear_tab(self, tab):
        for w in tab.winfo_children():
            w.destroy()
        self.figures = []

    def _build_summary_tab(self):
        self._clear_tab(self.tab_summary)
        container = ttk.Frame(self.tab_summary, padding=10)
        container.pack(fill="both", expand=True)

        # Header
        ttk.Label(container, text="Weak-Form Efficiency Results", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 10))

        # Search
        search_frame = ttk.Frame(container)
        search_frame.pack(fill="x", pady=(0, 10))
        search_var = tk.StringVar()
        ttk.Label(search_frame, text="Filter Stores: ").pack(side="left")
        ttk.Entry(search_frame, textvariable=search_var, width=20).pack(side="left", padx=5)
        ttk.Button(search_frame, text="Clear", command=lambda: search_var.set("")).pack(side="left")

        # Table
        cols = list(self.results_df.columns)
        tree_frame = ttk.Frame(container)
        tree_frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=20)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, anchor="center", width=120)
        tree.column('Store', width=80)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        def populate_table(filter_text=""):
            for i in tree.get_children():
                tree.delete(i)
            data = self.results_df
            if filter_text:
                data = data[data.apply(lambda r: filter_text.lower() in str(tuple(r.values)).lower(), axis=1)]
            for _, row in data.iterrows():
                tree.insert("", "end", values=[row[c] for c in cols])

        populate_table()
        search_var.trace_add("write", lambda *args: populate_table(search_var.get()))

        # Summary stats
        footer = ttk.Frame(container, padding=(0, 5))
        footer.pack(fill="x")
        eff_adf = (self.results_df['ADF Result'] == 'Efficient').sum()
        eff_runs = (self.results_df['Runs Result'] == 'Efficient').sum()
        ttk.Label(footer, text=f"Efficient Stores (ADF): {eff_adf}/{len(self.results_df)}").pack(side="left", padx=10)
        ttk.Label(footer, text=f"Efficient Stores (Runs): {eff_runs}/{len(self.results_df)}").pack(side="left")

    def _build_charts_tab(self):
        self._clear_tab(self.tab_charts)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        self.figures.append(fig)

        # Efficiency Counts
        adf_counts = self.results_df['ADF Result'].value_counts().reindex(['Efficient', 'Inefficient'], fill_value=0)
        sns.barplot(x=adf_counts.index, y=adf_counts.values, ax=axes[0, 0])
        axes[0, 0].set_title("ADF Test Results")
        axes[0, 0].set_ylabel("Number of Stores")

        runs_counts = self.results_df['Runs Result'].value_counts().reindex(['Efficient', 'Inefficient'], fill_value=0)
        sns.barplot(x=runs_counts.index, y=runs_counts.values, ax=axes[0, 1])
        axes[0, 1].set_title("Runs Test Results")
        axes[0, 1].set_ylabel("Number of Stores")

        # Monthly Sales
        monthly = self.df.groupby(self.df['Date'].dt.to_period('M'))['Weekly_Sales'].mean()
        monthly.index = monthly.index.to_timestamp()
        axes[1, 0].plot(monthly.index, monthly.values, marker='o', linewidth=1.5)
        axes[1, 0].set_title("Average Monthly Sales")
        axes[1, 0].set_xlabel("Date")
        axes[1, 0].set_ylabel("Sales ($)")
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # Store Average Sales
        store_avg = self.df.groupby('Store')['Weekly_Sales'].mean().sort_index()
        sns.barplot(x=store_avg.index, y=store_avg.values, ax=axes[1, 1])
        axes[1, 1].set_title("Average Sales by Store")
        axes[1, 1].set_xlabel("Store ID")
        axes[1, 1].set_ylabel("Sales ($)")
        axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        axes[1, 1].tick_params(axis='x', labelsize=8)

        fig.tight_layout(pad=2.0)
        canvas = FigureCanvasTkAgg(fig, master=self.tab_charts)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    def _build_acf_tab(self):
        self._clear_tab(self.tab_acf)
        top = ttk.Frame(self.tab_acf, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Select Store:", font=("Segoe UI", 10)).pack(side="left")
        stores = sorted(self.df['Store'].unique().tolist())
        self.sel_store = tk.IntVar(value=stores[0] if stores else 1)
        combo = ttk.Combobox(top, values=stores, textvariable=self.sel_store, width=10, state="readonly")
        combo.pack(side="left", padx=10)
        ttk.Button(top, text="Update Plot", command=self._update_acf_plot).pack(side="left")

        self.acf_container = ttk.Frame(self.tab_acf, padding=5)
        self.acf_container.pack(fill="both", expand=True)
        self._update_acf_plot()

    def _update_acf_plot(self):
        for w in self.acf_container.winfo_children():
            w.destroy()
        fig, ax = plt.subplots(figsize=(10, 4))
        self.figures.append(fig)

        s = int(self.sel_store.get())
        sub = self.df[self.df['Store'] == s].sort_values('Date').copy()
        sub['Returns'] = safe_pct_change(sub['Weekly_Sales'])
        returns = sub['Returns'].dropna()
        if len(returns) > 10:
            plot_acf(returns, lags=min(20, len(returns)-1), ax=ax)
            ax.set_title(f"Sales Returns Autocorrelation (Store {s})")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Autocorrelation")
        else:
            ax.text(0.5, 0.5, "Insufficient data for ACF plot", ha="center", va="center", fontsize=12)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.acf_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_reg_tab(self):
        self._clear_tab(self.tab_reg)
        container = ttk.Frame(self.tab_reg, padding=10)
        container.pack(fill="both", expand=True)

        # Regression Summary
        ttk.Label(container, text="Regression Analysis (Semi-Strong Form)", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 10))
        upper = ttk.Frame(container)
        upper.pack(fill="both", expand=True)
        txt = tk.Text(upper, wrap="word", height=16, font=("Consolas", 9))
        txt.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(upper, orient="vertical", command=txt.yview)
        sb.pack(side="right", fill="y")
        txt.configure(yscrollcommand=sb.set)
        txt.insert("1.0", model_summary_as_text(self.model))
        txt.config(state="disabled")

        # Diagnostic Plots
        lower = ttk.Frame(container, padding=(0, 10))
        lower.pack(fill="both", expand=True)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        self.figures.append(fig1)
        fitted = self.model.fittedvalues
        resid = self.model.resid
        sns.scatterplot(x=fitted, y=resid, ax=ax1, alpha=0.6)
        ax1.axhline(0, ls="--", lw=1, color="red")
        ax1.set_title("Residuals vs Fitted Values")
        ax1.set_xlabel("Fitted Values ($)")
        ax1.set_ylabel("Residuals")
        fig1.tight_layout()
        canvas1 = FigureCanvasTkAgg(fig1, master=lower)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side="left", fill="both", expand=True, padx=(0, 5))

        fig2 = plt.figure(figsize=(6, 4))
        self.figures.append(fig2)
        sm.qqplot(self.model.resid, line='45', fit=True, ax=plt.gca())
        plt.title("Q-Q Plot of Residuals")
        fig2.tight_layout()
        canvas2 = FigureCanvasTkAgg(fig2, master=lower)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side="left", fill="both", expand=True)

    def _build_extra_tab(self):
        self._clear_tab(self.tab_extra)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        self.figures.append(fig)

        # Correlation Heatmap
        corr_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        corr = self.df[corr_cols].corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", ax=axes[0, 0], cmap="coolwarm", cbar=True)
        axes[0, 0].set_title("Correlation Matrix")

        # Boxplot
        sns.boxplot(x='Holiday_Flag', y='Weekly_Sales', data=self.df, ax=axes[0, 1])
        axes[0, 1].set_title("Sales Distribution by Holiday Flag")
        axes[0, 1].set_xlabel("Holiday Flag")
        axes[0, 1].set_ylabel("Weekly Sales ($)")
        axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # Scatter Temperature
        sns.scatterplot(x='Temperature', y='Weekly_Sales', data=self.df, ax=axes[1, 0], alpha=0.6)
        axes[1, 0].set_title("Sales vs Temperature")
        axes[1, 0].set_xlabel("Temperature (°F)")
        axes[1, 0].set_ylabel("Weekly Sales ($)")
        axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # Scatter CPI
        sns.scatterplot(x='CPI', y='Weekly_Sales', data=self.df, ax=axes[1, 1], alpha=0.6)
        axes[1, 1].set_title("Sales vs CPI")
        axes[1, 1].set_xlabel("CPI")
        axes[1, 1].set_ylabel("Weekly Sales ($)")
        axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        fig.tight_layout(pad=2.0)
        canvas = FigureCanvasTkAgg(fig, master=self.tab_extra)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    def export_results(self):
        if self.results_df is None:
            return
        folder = filedialog.askdirectory(title="Select Export Folder")
        if not folder:
            return
        try:
            csv_path = os.path.join(folder, "efficiency_results.csv")
            xls_path = os.path.join(folder, "analysis_results.xlsx")
            txt_path = os.path.join(folder, "regression_summary.txt")
            self.results_df.to_csv(csv_path, index=False)
            with pd.ExcelWriter(xls_path) as writer:
                self.results_df.to_excel(writer, sheet_name="Efficiency", index=False)
                self.df.to_excel(writer, sheet_name="Raw_Data", index=False)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(model_summary_as_text(self.model))
            figdir = os.path.join(folder, "figures")
            os.makedirs(figdir, exist_ok=True)
            for i, fig in enumerate(self.figures, start=1):
                fig.savefig(os.path.join(figdir, f"plot_{i}.png"), bbox_inches="tight")
            messagebox.showinfo("Export", "Results exported successfully")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")

    def _set_status(self, text):
        self.status_var.set(text)

def main():
    root = tk.Tk()
    style = ttk.Style()
    style.configure("Treeview", rowheight=25)
    style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
    try:
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app = WalmartApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()