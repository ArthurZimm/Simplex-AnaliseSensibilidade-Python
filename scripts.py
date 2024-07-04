import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pulp

class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Otimização Linear com PuLP")
        self.root.geometry("1280x960")
        self.root.eval('tk::PlaceWindow . center')  
        self.create_scrollable_container()
        self.setup_initial_interface()

    def create_scrollable_container(self):
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def setup_initial_interface(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        initial_frame = ttk.Frame(self.scrollable_frame, padding="10")
        initial_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(initial_frame, text="Número de Variáveis:", font=('Helvetica', 12)).grid(row=0, column=0, sticky=tk.E, pady=5)
        self.num_vars_entry = ttk.Entry(initial_frame)
        self.num_vars_entry.grid(row=0, column=1, pady=5)

        ttk.Label(initial_frame, text="Número de Restrições:", font=('Helvetica', 12)).grid(row=1, column=0, sticky=tk.E, pady=5)
        self.num_constraints_entry = ttk.Entry(initial_frame)
        self.num_constraints_entry.grid(row=1, column=1, pady=5)

        setup_button = ttk.Button(initial_frame, text="Confirmar", command=self.setup_variables_constraints)
        setup_button.grid(row=2, column=0, columnspan=2, pady=10)

        initial_frame.columnconfigure(0, weight=1)
        initial_frame.columnconfigure(1, weight=1)

    def setup_variables_constraints(self):
        try:
            self.num_vars = int(self.num_vars_entry.get() or 0)
            self.num_constraints = int(self.num_constraints_entry.get() or 0)

            if self.num_vars <= 0 or self.num_constraints <= 0:
                raise ValueError

            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()

            var_frame = ttk.Frame(self.scrollable_frame, padding="10")
            var_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

            ttk.Label(var_frame, text="Coeficientes da Função Objetiva", font=('Helvetica', 14, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
            self.var_entries = []
            for i in range(self.num_vars):
                ttk.Label(var_frame, text=f"Coeficiente de x{i+1}:", font=('Helvetica', 12)).grid(row=i+1, column=0, sticky=tk.E, pady=5)
                var_entry = ttk.Entry(var_frame)
                var_entry.grid(row=i+1, column=1, pady=5)
                self.var_entries.append(var_entry)

            const_frame = ttk.Frame(self.scrollable_frame, padding="10")
            const_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

            ttk.Label(const_frame, text="Restrições", font=('Helvetica', 14, 'bold')).grid(row=0, column=0, columnspan=self.num_vars + 2, pady=10)
            self.constraint_entries = []
            self.constraint_rhs_entries = []
            self.constraint_sign_entries = []

            for i in range(self.num_constraints):
                row = i + 1
                constraint_entry = []
                for j in range(self.num_vars):
                    entry = ttk.Entry(const_frame, width=5)
                    entry.grid(row=row, column=j, pady=5)
                    constraint_entry.append(entry)

                sign_entry = ttk.Combobox(const_frame, values=["<=", ">=", "="], width=5)
                sign_entry.grid(row=row, column=self.num_vars, pady=5)
                sign_entry.current(0) 
                self.constraint_sign_entries.append(sign_entry)

                rhs_entry = ttk.Entry(const_frame, width=5)
                rhs_entry.grid(row=row, column=self.num_vars + 1, pady=5)
                self.constraint_rhs_entries.append(rhs_entry)

                self.constraint_entries.append(constraint_entry)

            solve_button = ttk.Button(self.scrollable_frame, text="Resolver", command=self.solve_optimization)
            solve_button.grid(row=2, column=0, pady=10)

            self.tableau_frame = ttk.Frame(self.scrollable_frame, padding="10")
            self.tableau_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

            self.next_step_button = ttk.Button(self.scrollable_frame, text="Próximo Passo", command=self.next_simplex_step)
            self.next_step_button.grid(row=4, column=0, pady=10)
            self.next_step_button.grid_remove()  

            self.basic_vars_text = tk.StringVar()
            basic_vars_label = ttk.Label(self.scrollable_frame, textvariable=self.basic_vars_text, justify="left", font=('Helvetica', 12))
            basic_vars_label.grid(row=5, column=0, sticky="nw")

            self.sensitivity_text = tk.StringVar()
            sensitivity_label = ttk.Label(self.scrollable_frame, textvariable=self.sensitivity_text, justify="left", font=('Helvetica', 12))
            sensitivity_label.grid(row=7, column=0, sticky="nw")  

            self.sensitivity_button = ttk.Button(self.scrollable_frame, text="Resolver Análise de Sensibilidade", command=self.sensitivity_analysis)
            self.sensitivity_button.grid(row=6, column=0, pady=10)
            self.sensitivity_button.grid_remove() 

            reset_button = ttk.Button(self.scrollable_frame, text="Reiniciar", command=self.setup_initial_interface)
            reset_button.grid(row=8, column=0, pady=10)

            self.scrollable_frame.columnconfigure(0, weight=1)

        except ValueError:
            messagebox.showerror("Erro de entrada", "Por favor, insira números válidos.")

    def solve_optimization(self):
        try:
            var_coefs = [float(entry.get() or 0) for entry in self.var_entries]

            constraints = []
            rhs_values = []
            signs = []

            for i in range(self.num_constraints):
                constraint = [float(entry.get() or 0) for entry in self.constraint_entries[i]]
                constraints.append(constraint)
                rhs_values.append(float(self.constraint_rhs_entries[i].get() or 0))
                signs.append(self.constraint_sign_entries[i].get())

            self.var_coefs = var_coefs
            self.constraints = constraints
            self.rhs_values = rhs_values
            self.signs = signs


            self.current_tableau = self.create_initial_tableau()
            self.show_tableau(self.current_tableau)
            self.next_step_button.grid() 

        except ValueError:
            messagebox.showerror("Erro de entrada", "Por favor, insira valores numéricos válidos.")

    def create_initial_tableau(self):
        num_constraints = self.num_constraints
        num_vars = self.num_vars

        tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))

        for i in range(num_constraints):
            tableau[i, :num_vars] = self.constraints[i]
            tableau[i, num_vars + i] = 1
            tableau[i, -1] = self.rhs_values[i]

        tableau[-1, :num_vars] = -np.array(self.var_coefs)

        return tableau

    def show_tableau(self, tableau, pivot_row=None, pivot_col=None):
        for widget in self.tableau_frame.winfo_children():
            widget.destroy()

        num_rows, num_cols = tableau.shape

        headers = [f'x{i+1}' for i in range(self.num_vars)] + [f's{i+1}' for i in range(self.num_constraints)] + ['RHS']
        for j, header in enumerate(headers):
            label = ttk.Label(self.tableau_frame, text=header, borderwidth=1, relief="solid", width=10, font=('Helvetica', 10, 'bold'))
            label.grid(row=0, column=j + 1, padx=1, pady=1)

        for i in range(num_rows):
            if i < num_rows - 1:
                label = ttk.Label(self.tableau_frame, text=f"s{i+1}", borderwidth=1, relief="solid", width=10, font=('Helvetica', 10, 'bold'))
            else:
                label = ttk.Label(self.tableau_frame, text="Z", borderwidth=1, relief="solid", width=10, font=('Helvetica', 10, 'bold'))
            label.grid(row=i + 1, column=0, padx=1, pady=1)

            for j in range(num_cols):
                value = f"{tableau[i, j]:.2f}"
                label = ttk.Label(self.tableau_frame, text=value, borderwidth=1, relief="solid", width=10)
                label.grid(row=i + 1, column=j + 1, padx=1, pady=1)
                if pivot_row is not None and pivot_col is not None and i == pivot_row and j == pivot_col:
                    label.configure(background="lightgreen")

    def next_simplex_step(self):
        tableau = self.current_tableau

        last_row = tableau[-1, :-1]
        pivot_col = np.argmin(last_row)
        if last_row[pivot_col] >= 0:
            self.next_step_button.config(state="disabled")
            self.display_basic_non_basic_vars()
            self.sensitivity_button.grid() 
            return

        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        valid_ratios = [ratios[i] if tableau[i, pivot_col] > 0 else float('inf') for i in range(len(ratios))]
        pivot_row = np.argmin(valid_ratios)

        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element

        for i in range(len(tableau)):
            if i != pivot_row:
                row_factor = tableau[i, pivot_col]
                tableau[i, :] -= row_factor * tableau[pivot_row, :]

        self.current_tableau = tableau
        self.show_tableau(tableau, pivot_row=pivot_row, pivot_col=pivot_col)

    def display_basic_non_basic_vars(self):
        tableau = self.current_tableau
        num_vars = self.num_vars
        num_constraints = self.num_constraints

        basic_vars = []
        non_basic_vars = []

        for i in range(num_constraints):
            if tableau[i, i + num_vars] == 1 and all(tableau[j, i + num_vars] == 0 for j in range(num_constraints) if j != i):
                basic_vars.append((f's{i+1}', tableau[i, -1]))
            else:
                for k in range(num_vars):
                    if tableau[i, k] == 1 and all(tableau[j, k] == 0 for j in range(num_constraints) if j != i):
                        basic_vars.append((f'x{k+1}', tableau[i, -1]))

        z_value = tableau[-1, -1]

        result = "Variáveis Básicas:\n"
        for var, value in basic_vars:
            result += f"{var} = {value:.2f}\n"
        result += "\nVariáveis Não Básicas:\n"
        for var in non_basic_vars:
            result += f"{var} = 0.00\n"
        result += f"\nValor de Z: {z_value:.2f}"

        self.basic_vars_text.set(result)

    def sensitivity_analysis(self):
        sensitivity_result = "Análise de Sensibilidade:\n"
        sensitivity_result += f"{'Restrição':<17} | {'Preço Sombra':<17} | {'Slack':<10}\n"

        var_coefs = self.var_coefs
        constraints = self.constraints
        rhs_values = self.rhs_values
        signs = self.signs

        problem = pulp.LpProblem("Maximizar_Lucro", pulp.LpMaximize)

        variables = [pulp.LpVariable(f'x{i+1}', lowBound=0, cat='Continuous') for i in range(self.num_vars)]
        problem += pulp.lpSum([var_coefs[i] * variables[i] for i in range(self.num_vars)]), "Profit"

        for i in range(self.num_constraints):
            if signs[i] == "<=":
                problem += pulp.lpSum([constraints[i][j] * variables[j] for j in range(self.num_vars)]) <= rhs_values[i]
            elif signs[i] == ">=":
                problem += pulp.lpSum([constraints[i][j] * variables[j] for j in range(self.num_vars)]) >= rhs_values[i]
            elif signs[i] == "=":
                problem += pulp.lpSum([constraints[i][j] * variables[j] for j in range(self.num_vars)]) == rhs_values[i]

        problem.solve()

        for name, constraint in problem.constraints.items():
            sensitivity_result += f"{name:<20} | {constraint.pi:<30.2f} | {constraint.slack:<10.2f}\n"

        sensitivity_result += "\nCustos Reduzidos (Reduced Costs):\n"
        sensitivity_result += f"{'Variável':<10} {'Custo Reduzido':<10}\n"
        for variable in problem.variables():
            sensitivity_result += f"{variable.name:<10} {variable.dj:<10.2f}\n"

        self.sensitivity_text.set(sensitivity_result)

root = tk.Tk()
app = OptimizationApp(root)
root.mainloop()
