import pandas as pd
import os
import sys

def main():
    # Nombres de archivos
    input_file = "resultados_finales_Q1.xlsx"
    output_file = "datos_para_chat.txt"

    # Verificar existencia del Excel
    if not os.path.exists(input_file):
        print(f"❌ ERROR: No encuentro el archivo '{input_file}' en este directorio.")
        print("   Asegúrate de haber ejecutado primero 'build_analisis_final_v4.py'.")
        return

    print(f"📂 Leyendo '{input_file}'...")
    
    try:
        xl = pd.ExcelFile(input_file)
    except Exception as e:
        print(f"❌ Error crítico abriendo el Excel: {e}")
        return

    # Verificar si tabulate está instalado para formato bonito, sino usar formato estándar
    try:
        import tabulate
        HAS_TABULATE = True
    except ImportError:
        HAS_TABULATE = False
        print("⚠️ Nota: 'tabulate' no está instalado. Las tablas saldrán en formato texto simple.")
        print("   (Para mejor formato: pip install tabulate)")

    # Hojas que nos interesan para el artículo
    sheets_of_interest = [
        ("BASELINE_METRICS", "TABLA 1: Resultados Globales (Accuracy/F1)"),
        ("BASELINE_MCNEMAR", "TABLA 2: Significancia Estadística (Exacta)"),
        ("BASELINE_DELTAS_CI", "TABLA 3: Intervalos de Confianza (Robustez)"),
        ("SENS_MCNEMAR_POOLED", "TABLA 4: Robustez Agregada (Sensibilidad)"),
        ("PRCC_INFERENCE", "TABLA 5: Análisis PRCC (Importancia de Parámetros)"),
        ("PARETO_FRONTS", "FIGURA/TABLA 6: Frentes de Pareto (Trade-offs)"),
        ("REPRODUCIBILITY", "ANEXO: Datos de Reproducibilidad"),
        ("MISSING_WARNINGS", "DEBUG: Advertencias (si las hay)")
    ]

    print(f"📝 Escribiendo resultados en '{output_file}'...")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== DATOS EXPORTADOS PARA ANÁLISIS DE ARTÍCULO CIENTÍFICO ===\n")
        f.write(f"Origen: {input_file}\n")
        f.write("Objetivo: Generación de secciones 'Resultados' y 'Discusión'\n")
        f.write("============================================================\n\n")

        found_count = 0

        for sheet_name, description in sheets_of_interest:
            f.write(f"\n{'='*80}\n")
            f.write(f"HOJA: {sheet_name}\n")
            f.write(f"INFO: {description}\n")
            f.write(f"{'='*80}\n\n")

            if sheet_name not in xl.sheet_names:
                f.write(f"[⚠️ LA HOJA '{sheet_name}' NO EXISTE EN EL EXCEL]\n")
                continue

            try:
                df = xl.parse(sheet_name)
                
                if df.empty:
                    f.write("[LA HOJA ESTÁ VACÍA]\n")
                    continue
                
                found_count += 1

                # Lógica de formateo
                # Si es muy larga (ej. Pareto crudo), mostramos resumen
                if len(df) > 100:
                    f.write(f"NOTA: La tabla tiene {len(df)} filas. Se muestran las primeras 40 y últimas 40.\n\n")
                    subset = pd.concat([df.head(40), df.tail(40)])
                    if HAS_TABULATE:
                        f.write(subset.to_markdown(index=False, tablefmt="grid"))
                    else:
                        f.write(subset.to_string(index=False))
                else:
                    if HAS_TABULATE:
                        f.write(df.to_markdown(index=False, tablefmt="grid"))
                    else:
                        f.write(df.to_string(index=False))
                
                f.write("\n\n")

            except Exception as e:
                f.write(f"[ERROR LEYENDO HOJA: {e}]\n")

    print(f"✅ ¡LISTO! Se ha generado el archivo: {output_file}")
    print(f"   Se procesaron {found_count} hojas correctamente.")
    print("👉 Por favor, sube el archivo 'datos_para_chat.txt' al chat.")

if __name__ == "__main__":
    main()