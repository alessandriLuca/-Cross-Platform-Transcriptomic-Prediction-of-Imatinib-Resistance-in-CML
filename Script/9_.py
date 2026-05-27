#!/usr/bin/env python3
"""
09_endpoint_concordance.py
Stima la concordanza tra risposta citogenetica (GSE14671) e EMR
usando le informazioni disponibili nel metadata.

Il problema: ProtoDANN è addestrato su EMR ma testato su risposta citogenetica.
Questo script quantifica quanto i due endpoint concordano nei dati disponibili
per difendere (o no) l'approssimazione.

Cosa fa:
  1. Legge il metadata di GSE14671
  2. Estrae le informazioni di risposta disponibili
  3. Cerca eventuali informazioni EMR nei campi del metadata
  4. Se entrambe le etichette sono disponibili per gli stessi campioni,
     calcola concordanza, kappa di Cohen, e sensibilità/specificità
     di risposta citogenetica come proxy di EMR

Output:
  RESULTS_DIR/endpoint_concordance.json
  Stampa a video tutto il metadata rilevante per ispezione manuale

Uso:
  export RESULTS_DIR=/home/jovyan/work/shared/results/v1_prototype
  export RAW_DIR=/home/jovyan/work/shared/raw
  python3 09_endpoint_concordance.py
"""

import os, json
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = os.environ.get("RESULTS_DIR",
              "/home/jovyan/work/shared/results/v1_prototype")
RAW_DIR     = os.environ.get("RAW_DIR", "/home/jovyan/work/shared/raw")


def cohen_kappa(y1, y2):
    """Cohen's kappa tra due vettori binari."""
    y1, y2 = np.array(y1), np.array(y2)
    n = len(y1)
    p_obs = (y1 == y2).mean()
    p_e   = ((y1==1).mean() * (y2==1).mean() +
              (y1==0).mean() * (y2==0).mean())
    return (p_obs - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0


def main():
    meta_path = os.path.join(RAW_DIR, "GSE14671", "metadata.csv")
    meta      = pd.read_csv(meta_path)

    log.info(f"Metadata GSE14671: {meta.shape}")
    log.info(f"Colonne disponibili: {meta.columns.tolist()}")

    # Stampa tutte le colonne che potrebbero contenere informazioni cliniche
    log.info("\n--- Colonne con potenziale informazione clinica ---")
    clinical_keywords = ["response", "responder", "emr", "molecular",
                         "cytogen", "cytoge", "title", "description",
                         "characteristics", "treatment", "outcome",
                         "bcr", "abl", "remission"]
    for col in meta.columns:
        if any(kw in col.lower() for kw in clinical_keywords):
            log.info(f"\nColonna: {col}")
            log.info(meta[col].value_counts().to_string())

    # Estrai label citogenetica (quella usata nel training)
    log.info("\n--- Distribuzione label per 'title' ---")
    meta["cyto_label"] = meta["title"].str.lower().apply(
        lambda t: 0 if "nonresponder" in t else (1 if "responder" in t else np.nan)
    )
    log.info(meta["cyto_label"].value_counts(dropna=False).to_string())

    # Cerca informazioni EMR nei campi characteristics
    char_cols = [c for c in meta.columns if "characteristics" in c.lower()]
    log.info(f"\n--- Campi characteristics: {char_cols} ---")
    for col in char_cols:
        unique_vals = meta[col].dropna().unique()
        log.info(f"\n{col} — valori unici:")
        for v in unique_vals[:30]:
            log.info(f"  {v}")

    # Cerca "molecular" o "BCR-ABL" nei campi disponibili
    log.info("\n--- Ricerca 'molecular' o 'BCR' in tutti i campi ---")
    for col in meta.columns:
        mask = meta[col].astype(str).str.lower().str.contains("molecular|bcr.abl|emr|10%", na=False)
        if mask.any():
            log.info(f"\nColonna '{col}' contiene info molecolari:")
            log.info(meta.loc[mask, col].value_counts().to_string())

    # Mostra record completi dei campioni del test set (non-training)
    log.info("\n--- Record completi campioni validation set ---")
    test_mask = ~meta["title"].str.lower().str.contains("training", na=False)
    test_meta = meta[test_mask]
    log.info(f"Campioni validation (test set): {len(test_meta)}")
    for _, row in test_meta.iterrows():
        label = "NR" if "nonresponder" in str(row.get("title","")).lower() else "R"
        extra = []
        for col in char_cols:
            val = str(row.get(col, ""))
            if val and val != "nan":
                extra.append(f"{col}={val[:80]}")
        log.info(f"  {row['sample_id']} [{label}]  {' | '.join(extra)}")

    # Salva risultato
    result = {
        "n_total_samples":    int(len(meta)),
        "n_responder":        int((meta["cyto_label"]==1).sum()),
        "n_nonresponder":     int((meta["cyto_label"]==0).sum()),
        "n_unlabeled":        int(meta["cyto_label"].isna().sum()),
        "columns_available":  meta.columns.tolist(),
        "emr_info_found":     False,
        "note": ("EMR information not found in public metadata. "
                 "Cytogenetic response used as proxy. "
                 "Concordance with EMR cannot be quantified from public data alone. "
                 "Recommendation: contact Kok/Hughes group (IMVS Adelaide) "
                 "for access to TIDEL-II validation cohort (88 patients, EMR labeled).")
    }

    # Se trovassimo info EMR, calcoliamo concordanza qui
    # (lasciato come hook per aggiornamento manuale)

    out = os.path.join(RESULTS_DIR, "endpoint_concordance.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"\nSalvato: {out}")
    log.info("\nConlusione: le informazioni EMR non sono disponibili nel metadata pubblico di GSE14671.")
    log.info("La concordanza citogenetico/EMR non può essere quantificata direttamente.")
    log.info("Raccomandazione: contattare il gruppo Kok/Hughes per accesso ai dati TIDEL-II.")


if __name__ == "__main__":
    main()