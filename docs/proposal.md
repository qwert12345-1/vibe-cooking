# Ingredient-Based Recipe Recommender: Project Proposal

**Problem.** We propose to build an interactive web system that recommends cooking
recipes based on the ingredients a user currently has on hand. Our dataset is
[`datahiveai/recipes-with-nutrition`](https://huggingface.co/datasets/datahiveai/recipes-with-nutrition)
on Hugging Face, which contains 39,447 recipes scraped from major cooking sites
(Food Network, Serious Eats, Food52, Cookstr, and others). Each record provides a
structured ingredient list with normalized food names (e.g.,
`{"food": "green cabbage", "weight": 280, "measure": "cup"}`), cuisine labels
(e.g., `"american"`, `"south east asian"`), diet and health tags (e.g.,
`"Vegetarian"`, `"Gluten-Free"`, `"Low-Carb"`), meal type, recipe URL, image, and
a full nutritional breakdown. The user-facing question is simple and practically
useful: *"Given what's in my fridge right now, what should I cook tonight?"*
This is interesting both as a consumer problem — it addresses food waste and the
everyday "what's for dinner" decision-fatigue that applies to nearly anyone who
cooks — and as a machine-learning problem. Real user input is noisy (typos like
*"chiken"*, morphological variants like *"tomatoes"* vs *"tomato"*, foreign-language
input like *"洋葱"*, and abstract superordinate terms like *"poultry"*); the
relevance signal is multi-dimensional (pantry staples such as salt and water
should not dominate matching, while rare ingredients should); and users expect
to layer soft preferences on top of hard ingredient constraints (*"Southeast
Asian, spicy, no beef, under 500 calories"*). A system that handles these
realities well is substantially more useful than a literal keyword search.

**Methods.** Our pipeline combines three core algorithms from the course, two of
which we will implement from scratch with numpy/scipy rather than calling a
library. First, we will hand-implement **TF-IDF vectorization with cosine
similarity** as the main retrieval engine: each recipe becomes a sparse vector
over an ingredient vocabulary built from the dataset, and the user's ingredient
list is scored against every recipe using cosine similarity. TF-IDF is
appropriate because its inverse-document-frequency term automatically
down-weights pantry staples (salt, water, oil) that appear in most recipes
while up-weighting distinctive ingredients (saffron, miso, mascarpone) — which
is exactly the matching behavior we want for this task. Second, we will
hand-implement **K-Means clustering** (with k-means++ initialization and
silhouette-based selection of k) on sentence-embedding representations of recipes to
produce an interactive 2-D "recipe universe" visualization, letting the user
click a recommended recipe and explore its neighbors within the same cluster.
K-Means is appropriate because the recipe space has natural groupings (soups,
baked goods, stir-fries, desserts, drinks) and cluster structure gives users an
intuitive way to browse beyond the top-K list. To make the system robust to
noisy input, we wrap the retrieval core in a multi-stage input-normalization
pipeline that combines **lemmatization**, **Levenshtein-based fuzzy matching**,
and **sentence-transformer semantic similarity** as progressive fallbacks when
exact vocabulary matching fails, with an interactive confirmation UI that
surfaces low-confidence matches for the user to accept, reject, or replace. The
complete system will be served through a Gradio web interface with faceted
filters (cuisine, diet, calorie range, excluded ingredients, cooking style)
applied as hard pre-filters before retrieval, plus a Plotly-based cluster
visualization for exploration.
