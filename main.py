import os
import numpy as np
import pandas as pd

import fasterrcnn_crop as smart_cropper
import classifier_manager
import extractor_resnet_imagenet as resnet_imagenet
import extractor_vit as vit
import extractor_resnet_custom
import dbscan_manager as dbscan 

# --- CONFIGURAZIONE ---
IMAGE_FOLDER = "../dataset"
EXCEL_FILE = "my_dataset.xlsx"

CROPPED_FOLDER = "../dataset_new_cropper"
DISCARD_FOLDER = "../dataset_discarded"

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
    print("3. ResNet18 CUSTOM")
    print("0. Annulla e torna al menu principale")
    scelta = input("Scelta: ")
    
    if scelta == "0":
        return None

    print("\nQuale dataset vuoi analizzare?")
    print("1. Dataset Originale")
    print("2. Dataset Ritagliato (Smart Cropping)")
    scelta_ds = input("Scelta (1/2): ")

    if scelta_ds == "2":
        target_folder = CROPPED_FOLDER
        prefisso = "cropped_"
    else:
        target_folder = IMAGE_FOLDER
        prefisso = "base_"
    
    if not os.path.exists(target_folder):
        print(f"\n[!] Cartella '{target_folder}' non trovata.")
        return None 
    
    output_file = ""
    
    if scelta == "1":
        print(f"\nAvvio ResNet addestrato su Imagenet (Target: {tag_gruppo})...")
        model = resnet_imagenet.get_model()
        transform = resnet_imagenet.get_transforms()
        
        embeddings, names = resnet_imagenet.run_extraction(target_folder, model, transform, valid_names=nomi_validi)
        
        output_file = f"features_resnet18_imagenet_{prefisso}{tag_gruppo}.npz"
        np.savez(output_file, embeddings=embeddings, names=np.array(names))
        
    elif scelta == "2":
        print(f"\nAvvio ViT (Target: {tag_gruppo})...")
        model = vit.get_model()
        transform = vit.get_transforms()
        
        embeddings, names = vit.run_extraction(target_folder, model, transform, valid_names=nomi_validi)
        
        output_file = f"features_vit_{prefisso}{tag_gruppo}.npz"
        np.savez(output_file, embeddings=embeddings, names=np.array(names))

    elif scelta == "3":
        print(f"\nAvvio ResNet18 CUSTOM (Target: {tag_gruppo})...")
        output_file = f"features_resnet_custom_{prefisso}{tag_gruppo}.npz"
        
        extractor_resnet_custom.extract_features(target_folder, nomi_validi, output_file)
    
    else:
        print("\n[!] Scelta non valida.")
        return None

    if output_file:
        print(f"\n[✓] Estrazione completata! File creato: {output_file}")
    
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
        print("0. PRE-PROCESSING (Smart Cropping)")
        print("1. ESTRAZIONE FEATURE (ResNet_imagenet o ViT)")
        print("2. ANALISI E CLUSTERING (DBSCAN Manager)")
        print("3. CAMBIA GRUPPO / FILTRO IMMAGINI")
        print("4. CLASSIFICAZIONE SUPERVISIONATA (ResNet Custom)")
        print("00. ESCI")
        
        scelta_principale = input("\nSeleziona fase: ")

        if scelta_principale == "0":
            smart_cropper.run_smart_cropping(IMAGE_FOLDER, CROPPED_FOLDER, DISCARD_FOLDER, valid_names=nomi_validi)

        elif scelta_principale == "1":
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

        elif scelta_principale == "4":
            print("\n--- CLASSIFICAZIONE IMMAGINI ---")
            print("Quale dataset vuoi analizzare?")
            print("1. Dataset Originale")
            print("2. Dataset Ritagliato (Smart Cropping)")
            scelta_ds = input("Scelta (1/2): ")

            if scelta_ds == "2":
                target_folder = CROPPED_FOLDER
                nome_file = f"risultati_classificazione_cropped_{tag_gruppo}.xlsx"
            else:
                target_folder = IMAGE_FOLDER
                nome_file = f"risultati_classificazione_base_{tag_gruppo}.xlsx"

            if not os.path.exists(target_folder):
                print(f"\n[!] Cartella '{target_folder}' non trovata. Devi prima eseguire il cropping?")
                continue

            modello_class = classifier_manager.get_model()
            if modello_class is not None:
                transf_class = classifier_manager.get_transforms()
                classifier_manager.run_classification(
                    folder=target_folder, 
                    model=modello_class, 
                    transform=transf_class, 
                    valid_names=nomi_validi, 
                    output_xlsx=nome_file,
                    excel_path=EXCEL_FILE
                )
        
        elif scelta_principale == "00":
            print("Chiusura progetto.")
            break
        else:
            print("Opzione non valida.")

if __name__ == "__main__":
    main()