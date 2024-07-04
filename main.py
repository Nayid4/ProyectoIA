import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.data_processing import load_data, clean_data
from src.model import train_model, predict

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Aplicación de IA para Vinos")
        self.master.geometry("1100x600")
        self.create_widgets()

    def create_widgets(self):
        # Contenedor para botones y formulario
        left_frame = tk.Frame(self.master, padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.title_label = tk.Label(left_frame, text="IA para Vinos", font=("Helvetica", 24, "bold"), fg='black')
        self.title_label.pack(pady=10)

        # Contenedor para los botones de carga y entrenamiento
        button_container = tk.Frame(left_frame)
        button_container.pack(pady=10)

        # Botones
        self.load_button = tk.Button(button_container, text="Cargar Dataset", command=self.load_data, width=20, bg='black', fg='white')
        self.load_button.grid(row=0, column=0, padx=5)

        self.train_button = tk.Button(button_container, text="Entrenar Modelo", command=self.train_model, width=20, bg='black', fg='white')
        self.train_button.grid(row=0, column=1, padx=5)

        # Formulario
        self.test_frame = tk.Frame(left_frame, pady=20)
        self.test_frame.pack()

        self.type_label = tk.Label(self.test_frame, text="Tipo de Vino:")
        self.type_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.type_combo = ttk.Combobox(self.test_frame, values=['white', 'red'])
        self.type_combo.grid(row=0, column=1, padx=10, pady=5)

        self.features = [
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol"
        ]

        self.entries = {}
        for i, feature in enumerate(self.features, start=1):
            label = tk.Label(self.test_frame, text=feature + ":")
            label.grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
            entry = tk.Entry(self.test_frame)
            entry.grid(row=i, column=1, padx=10, pady=5)
            self.entries[feature] = entry

        self.predict_button = tk.Button(self.test_frame, text="Probar Modelo", command=self.predict_model, width=20, bg='black', fg='white')
        self.predict_button.grid(row=len(self.features) + 1, columnspan=2, pady=10)

        self.result_label = tk.Label(self.test_frame, text="")
        self.result_label.grid(row=len(self.features) + 2, columnspan=2, pady=10)

        # Contenedor para la gráfica
        right_frame = tk.Frame(self.master)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df = load_data(file_path)
            self.df = clean_data(self.df)
            messagebox.showinfo("Información", "Dataset cargado y limpiado correctamente")

    def train_model(self):
        if hasattr(self, 'df'):
            self.model, self.mse, self.r2, self.y_test, self.y_pred, self.iter_errors = train_model(self.df)
            self.update_error_plot()
            messagebox.showinfo("Resultados", f'MSE: {self.mse}\nR2: {self.r2}')
        else:
            messagebox.showwarning("Advertencia", "Primero debe cargar el dataset")

    def update_error_plot(self):
        self.figure.clear()

        if hasattr(self, 'iter_errors'):
            iterations = range(1, len(self.iter_errors) + 1)
            plt.plot(iterations, self.iter_errors, marker='o', linestyle='-', color='b', label='Error de Iteración (MSE)')
            plt.xlabel('Iteración')
            plt.ylabel('Error')
            plt.title('Error de Iteración del Modelo')
            plt.legend()
        else:
            messagebox.showwarning("Advertencia", "Primero debe entrenar el modelo y obtener los datos de iteración")

        self.canvas.draw()

    def predict_model(self):
        if not self.type_combo.get():
            messagebox.showwarning("Advertencia", "Seleccione el tipo de vino")
            return

        for feature in self.features:
            if not self.entries[feature].get():
                messagebox.showwarning("Advertencia", f"El campo {feature} no puede estar vacío")
                return

        if hasattr(self, 'model'):
            data = [1 if self.type_combo.get() == 'red' else 0]
            try:
                data.extend(float(self.entries[feature].get()) for feature in self.features)
            except ValueError:
                messagebox.showwarning("Advertencia", "Todos los campos deben contener valores numéricos válidos")
                return

            result = predict(self.model, data)
            self.result_label.config(text=f"Predicción de calidad: {result:.2f}", fg='black')
        else:
            messagebox.showwarning("Advertencia", "Primero debe entrenar el modelo")

root = tk.Tk()
app = Application(master=root)
app.mainloop()
