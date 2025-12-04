#!/bin/bash

set -e

ask_for_confirmation() {
    read -p "$1 (y/n): " response
    if [[ "$response" != "y" && "$response" != "Y" ]]; then
        echo "[INFO] Aborting."
        exit 1
    fi
}

ask_for_confirmation "Do you want to delete existing [documents] and [chunks] table and regenerate? [Y/N]"
echo "[INFO] regenerating dataset db for documents and chunks"


python ./db/gen_dataset_db.py \
    --regen True \
    --docs_path ./dragonball_dataset/dragonball_docs.jsonl
echo "[INFO] All dataset db regenerated."

# ask_for_confirmation "Do you want to delete existing [queries] table and regenerate? [Y/N]"
# echo "[INFO] regenerating dataset db for queries"

# python ./db/gen_query_db.py \
#     --regen True \
#     --docs_path ./dragonball_dataset/dragonball_queries.jsonl
# echo "[INFO] All query db regenerated."
