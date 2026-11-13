#!/usr/bin/env bash
# Elenca (NON cancella) gli output prodotti prima della riorganizzazione in
# derivatives/laminar_psc/<metodo>/0X_<script>/.
#
# Le vecchie cartelle di output sono obsolete per intero (anche le eventuali
# sottocartelle equidist/ equivol/ *_voxelwise/ della versione intermedia).
# La cache delle mappe PSC invece va SPOSTATA, non cancellata: lo script stampa
# il comando mv in fondo.
#
# Uso:  bash tools/list_old_outputs.sh
# Scrive la lista in old_outputs.txt; dopo averla controllata:
#       xargs -d '\n' rm -v < old_outputs.txt
#       find <cartelle vecchie> -type d -empty -delete

ROOT1=/mnt/store1/projects/progetto_layer/data/derivatives
ROOT2=/mnt/store2/projects/progetto_layer/data/derivatives   # tmap_subject.py usa store2
LISTA=old_outputs.txt
: > "$LISTA"

sezione () {   # $1 = titolo, resto = comando find
  local titolo=$1; shift
  echo; echo "=== $titolo"
  local out
  out=$("$@" 2>/dev/null | sort)
  if [ -z "$out" ]; then echo "  (niente)"; return; fi
  while IFS= read -r f; do
    ls -l --time-style=+%Y-%m-%d -- "$f" | awk '{printf "  %s  %8s  %s\n", $6, $5, $7}'
    echo "$f" >> "$LISTA"
  done <<< "$out"
}

# --- vecchie cartelle di output, obsolete per intero ----------------------------
sezione "group_layer_modelli/  (export_profiles.py + modelli_laminari.r)" \
  find "$ROOT1/group_layer_modelli" -type f
sezione "group_layer_ffb/  (layer_ff_fb_analysis.py)" \
  find "$ROOT1/group_layer_ffb" -type f
sezione "group_layer_parametric_psc/  (group_layer_fitting.py e _psc.py, equidist/equivol mescolati)" \
  find "$ROOT1/group_layer_parametric_psc" -type f
sezione "group_layer_plotting_psc/  (group_layer_plotting_psc.py)" \
  find "$ROOT1/group_layer_plotting_psc" -type f

# --- cache roi_scalar, non piu' usata -------------------------------------------
sezione "psc_maps: mappe del solo numeratore (modalita' roi_scalar, eliminata)" \
  find "$ROOT1/psc_maps" "$ROOT1/laminar_psc/psc_maps" -maxdepth 2 -type f -name '*_pscnum*'

# --- script legacy su spmT sogliati ---------------------------------------------
sezione "legacy/tmap_group.py: figure senza metrica nel nome" \
  find "$ROOT1/group_layer_plotting" -maxdepth 1 -type f -name '*.png' ! -name '*_equidist.png' ! -name '*_equivol.png'
sezione "legacy/tmap_subject.py (store2): figure senza metrica nel nome" \
  find "$ROOT2/layer_plotting" -maxdepth 2 -type f -name '*.png' ! -name '*_equidist.png' ! -name '*_equivol.png'

# --- altri file di profili sparsi, per sicurezza --------------------------------
echo; echo "=== Altri file *profil* fuori dalle cartelle sopra (solo da controllare, NON aggiunti alla lista)"
find "$ROOT1" -maxdepth 3 -type f \( -iname '*profil*.csv' -o -iname '*profil*.json' \) \
  ! -path '*/group_layer_modelli/*' ! -path '*/group_layer_ffb/*' \
  ! -path '*/group_layer_parametric_psc/*' ! -path '*/group_layer_plotting_psc/*' \
  ! -path '*/laminar_psc/*' 2>/dev/null | sed 's/^/  /'

# --- cache PSC: da SPOSTARE -----------------------------------------------------
echo; echo "=== Cache delle mappe PSC: da TENERE e spostare (non dipende dalla metrica)"
if [ -d "$ROOT1/psc_maps" ]; then
  du -sh "$ROOT1/psc_maps" | sed 's/^/  /'
  echo "  Spostala nella nuova struttura (dopo aver cancellato i *_pscnum*):"
  echo "    mkdir -p $ROOT1/laminar_psc && mv $ROOT1/psc_maps $ROOT1/laminar_psc/psc_maps"
elif [ -d "$ROOT1/laminar_psc/psc_maps" ]; then
  echo "  gia' in $ROOT1/laminar_psc/psc_maps"
else
  echo "  (non trovata: verra' ricalcolata al primo lancio)"
fi

echo; echo "Totale file vecchi: $(wc -l < "$LISTA")  -> lista in $LISTA"
echo "Per cancellarli dopo averli controllati:  xargs -d '\\n' rm -v < $LISTA"
