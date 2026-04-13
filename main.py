import os
import numpy as np
import pandas as pd

import extractor_resnet_imagenet as resnet_imagenet
import extractor_vit as vit
import dbscan_manager as dbscan 

# --- CONFIGURAZIONE ---
IMAGE_FOLDER = "../dataset" 
EXCEL_FILE = "my_dataset.xlsx"

def selezione_foto(excel_path=EXCEL_FILE):
    print("\nLettura dei filtri disponibili dal dataset...")
    try:
        df = pd.read_excel(excel_path)
        df.columns = df.columns.str.strip() 

        categorie = df['datasetCategory'].dropna().unique().tolist()
        annotazioni = df['personalAnnotation'].dropna().unique().tolist()

        mappa_opzioni = []

        print("\n" + "-"*50)
        print("SELEZIONE FILTRO IMMAGINI")
        print("-" * 50)
        
        indice = 1
        
        print("--- Filtra per Categoria ---")
        for cat in categorie:
            conteggio = len(df[df['datasetCategory'] == cat])
            print(f"[{indice}] Categoria: '{cat}' ({conteggio} immagini)")
            mappa_opzioni.append(('datasetCategory', cat))
            indice += 1

        print("\n--- Filtra per Annotazione ---")
        for ann in annotazioni:
            conteggio = len(df[df['personalAnnotation'] == ann])
            print(f"[{indice}] Annotazione: '{ann}' ({conteggio} immagini)")
            mappa_opzioni.append(('personalAnnotation', ann))
            indice += 1

        print("\n[0] Usa TUTTO il dataset (Nessun filtro)")
        print("-" * 50)

        while True:
            scelta = input("\nQuale gruppo vuoi elaborare?: ")
            
            if not scelta.isdigit():
                print("Inserisci un numero valido.")
                continue
                
            scelta = int(scelta)

            if scelta == 0:
                print("-> Selezionato: INTERO DATASET")
                return None, "ALL_DATA"

            if 1 <= scelta <= len(mappa_opzioni):
                colonna_scelta, valore_scelto = mappa_opzioni[scelta - 1]

                mask = df[colonna_scelta] == valore_scelto
                nomi_validi = set(df[mask]['image name'].astype(str).tolist())
                
                tag_gruppo = str(valore_scelto).replace(" ", "_")
                
                print(f"-> Selezionato: {colonna_scelta} = '{valore_scelto}' ({len(nomi_validi)} immagini valide trovate)")
                return nomi_validi, tag_gruppo
            else:
                print("Scelta non valida. Riprova.")
                
    except Exception as e:
        print(f"Errore nella lettura dell'Excel: {e}")
        return None, "ERROR"

def menu_estrazione(nomi_validi, tag_gruppo):
    print("\n--- FASE 1: ESTRAZIONE FEATURE ---")
    print("1. Usa ResNet addestrato su Imagenet (512 features)")
    print("2. Usa Vision Transformer (ViT) (768 features)")
    print("0. Annulla e torna al menu principale")
    scelta = input("Scelta: ")
    
    output_file = ""
    if scelta == "1":
        print(f"\nAvvio ResNet addestrato su Imagenet (Target: {tag_gruppo})...")
        model = resnet_imagenet.get_model()
        transform = resnet_imagenet.get_transforms()
        
        embeddings, names = resnet_imagenet.run_extraction(IMAGE_FOLDER, model, transform, valid_names=nomi_validi)
        
        output_file = f"features_resnet18_imagenet_{tag_gruppo}.npz"
        np.savez(output_file, embeddings=embeddings, names=np.array(names))
        
    elif scelta == "2":
        print(f"\nAvvio ViT (Target: {tag_gruppo})...")
        model = vit.get_model()
        transform = vit.get_transforms()
        
        embeddings, names = vit.run_extraction(IMAGE_FOLDER, model, transform, valid_names=nomi_validi)
        
        output_file = f"features_vit_{tag_gruppo}.npz"
        np.savez(output_file, embeddings=embeddings, names=np.array(names))
    
    if output_file:
        print(f"\nEstrazione completata! File creato: {output_file}")
    
    return output_file

def main():
    nomi_validi, tag_gruppo = selezione_foto(EXCEL_FILE)
    
    if tag_gruppo == "ERROR":
        print("Impossibile avviare il sistema senza dataset. Chiusura.")
        return

    while True:
        print("\n" + "="*50)
        print(f"  SISTEMA ANALISI ORCHIDEE | SESSIONE: [{tag_gruppo}]")
        print("="*50)
        print("1. ESTRAZIONE FEATURE (ResNet_imagenet o ViT)")
        print("2. ANALISI E CLUSTERING (DBSCAN Manager)")
        print("3. CAMBIA GRUPPO / FILTRO IMMAGINI")
        print("0. ESCI")
        
        scelta_principale = input("\nSeleziona fase: ")

        if scelta_principale == "1":
            menu_estrazione(nomi_validi, tag_gruppo)
            
        elif scelta_principale == "2":
            dbscan.main(tag_gruppo) 
            
        elif scelta_principale == "3":
            print("\nAvvio procedura di cambio sessione...")
            nuovi_nomi, nuovo_tag = selezione_foto(EXCEL_FILE)
            
            if nuovo_tag != "ERROR":
                nomi_validi = nuovi_nomi
                tag_gruppo = nuovo_tag
                print(f"\n[✓] Sessione aggiornata con successo! Nuovo target: {tag_gruppo}")
            else:
                print("\n[!] Cambio sessione annullato o fallito. Mantenuta la sessione precedente.")

        elif scelta_principale == "0":
            print("Chiusura progetto.")
            break
        else:
            print("Opzione non valida.")

if __name__ == "__main__":
    main()