import os

# Configuración
extensiones_a_incluir = ['.py', '.json', '.html', '.css', '.js', '.md', '.txt']
carpetas_a_ignorar = ['__pycache__', 'venv', '.git', '.idea', '.vscode', 'node_modules']
nombre_archivo_salida = 'proyecto_completo_para_analizar.txt'

def unir_archivos():
    with open(nombre_archivo_salida, 'w', encoding='utf-8') as salida:
        # Recorrer la carpeta actual y subcarpetas
        for raiz, directorios, archivos in os.walk('.'):
            # Filtrar carpetas ignoradas
            directorios[:] = [d for d in directorios if d not in carpetas_a_ignorar]
            
            for archivo in archivos:
                ext = os.path.splitext(archivo)[1]
                if ext in extensiones_a_incluir and archivo != os.path.basename(__file__) and archivo != nombre_archivo_salida:
                    ruta_completa = os.path.join(raiz, archivo)
                    
                    try:
                        with open(ruta_completa, 'r', encoding='utf-8') as f:
                            contenido = f.read()
                            
                        # Escribir separador y nombre del archivo
                        salida.write(f"\n{'='*50}\n")
                        salida.write(f"ARCHIVO: {ruta_completa}\n")
                        salida.write(f"{'='*50}\n")
                        salida.write(contenido + "\n")
                        print(f"Agregado: {ruta_completa}")
                        
                    except Exception as e:
                        print(f"No se pudo leer {ruta_completa}: {e}")

    print(f"\n¡Listo! Sube el archivo '{nombre_archivo_salida}' al chat.")

if __name__ == '__main__':
    unir_archivos()