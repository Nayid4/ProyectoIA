
# src/visualization.py

import matplotlib.pyplot as plt

def plot_predictions_vs_real(y_test, y_pred, canvas):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.3)
    ax.set_xlabel('Valores Reales')
    ax.set_ylabel('Predicciones')
    ax.set_title('Predicciones vs Valores Reales')
    canvas.figure = fig
    canvas.draw()
