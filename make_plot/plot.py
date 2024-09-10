import matplotlib.pyplot as plt

# Funzione per leggere i vettori da un file di testo
def read_vector_from_file(filename):
    with open(filename, 'r') as file:
        # Legge tutti i valori e li converte in una lista di numeri
        vector = [float(line.strip()) for line in file]
    return vector

# Funzione per creare il grafico con due vettori
def plot_two_vectors(vector1, vector2):
    # Genera gli indici per l'asse delle ascisse (da 0 a n-1)
    x_values = range(len(vector1))
    
    # Crea il grafico
    plt.figure(figsize=(8, 6))
    
    # Traccia il primo vettore
    plt.plot(x_values, vector1, marker='o', linestyle='-', color='b', label='Generator Loss')
    
    # Traccia il secondo vettore
    plt.plot(x_values, vector2, marker='o', linestyle='--', color='r', label='Validation Generator Loss')
   
    #plt.ylim(0, 1.5)

    
    # Aggiunge etichette e titolo
    plt.xlabel('Epochs')
    plt.ylabel('G_Loss vs Val_G_Loss')
    plt.title('Generator Loss vs Validation Generator Loss in {} Epochs'.format(len(x_values)))
    
    # Mostra la legenda per distinguere i due vettori
    plt.legend()
    
    # Mostra la griglia
    plt.grid(True)
    
    # Mostra il grafico
    plt.show()

# Esempio di utilizzo
vector1 = read_vector_from_file('g_print.txt')  # Legge il primo vettore dal file "file1.txt"
vector2 = read_vector_from_file('vg_print.txt')  # Legge il secondo vettore dal file "file2.txt"

# Plot dei due vettori
plot_two_vectors(vector1, vector2)
