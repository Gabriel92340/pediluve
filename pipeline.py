# pipeline.py

from transformers import pipeline
import gradio as gr

# === 1️⃣ Chargement des modèles ===
# Tous les modèles choisis sont adaptés au français.
# Ils se téléchargent une seule fois, puis restent en cache.

summarizer = pipeline(
    "summarization",
    model="plguillou/t5-base-fr-sum-cnndm"
)

classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli"
)

ner = pipeline(
    "ner",
    model="Jean-Baptiste/camembert-ner",
    aggregation_strategy="simple"
)

# === 2️⃣ Fonction d’enrichissement du texte ===
def enrich_text(text):
    # Étape 1 : Résumé
    summary_output = summarizer(
        text,
        max_length=80,
        min_length=20,
        do_sample=False
    )
    summary = summary_output[0]["summary_text"]

    # Étape 2 : Classification
    candidate_labels = [
        "politique", "économie", "culture", "sport", "technologie", "santé", "éducation", "environnement"
    ]
    classification_output = classifier(text, candidate_labels)

    # Étape 3 : Extraction d'entités
    entities_output = ner(text)

    # 👉 Étape 4 : Retourner trois éléments distincts (un par sortie Gradio)
    return summary, classification_output, entities_output



# === 3️⃣ Interface utilisateur avec Gradio ===
demo = gr.Interface(
    fn=enrich_text,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Collez ici un article ou un paragraphe à analyser..."
    ),
    outputs=[
        gr.Textbox(label="Résumé généré"),
        gr.JSON(label="Classification"),
        gr.JSON(label="Entités nommées détectées")
    ],
    title="Pipeline d’Enrichissement de Texte",
    description=(
        "Cette application prend un texte en français et l’enrichit en trois étapes : "
        "résumé automatique, classification thématique, et détection d’entités nommées."
    )
)

# === 4️⃣ Lancer le programme ===
if __name__ == "__main__":
    demo.launch()
