---
license: cc-by-nc-4.0
language:
- en
pretty_name: 'Recipes With Nutrition '
---

Leverage our **Recipes Dataset** to explore one of the most comprehensive collections of structured recipe and nutrition data.  

This dataset contains **39,447 recipes**, enriched with detailed nutritional information, ingredient breakdowns, dietary classifications, and recipe metadata. Each record includes both raw recipe text (such as ingredient lines) and structured fields (nutritional values, cuisine type, diet/health labels, and more).  

Designed as a **rich, high-quality resource**, this dataset supports machine learning pipelines, recommendation engines, dietary research, nutritional analysis, and culinary trend studies.  

The dataset is delivered in **CSV format**, with JSON-stringified arrays and objects for complex fields, making it easy to integrate into analytics platforms, ML models, or databases.  

## Dataset Description  

- **Access:** Full dataset (39,447 recipes + structured nutritional/ingredient data)  
- **Curated by:** [DataHive](https://datahive.ai)  
- **Language(s):** English  
- **License:** Creative Commons Non-Commercial 4.0 (CC BY-NC 4.0)  

## Uses  

- **Nutritional Research** – Analyze macronutrient and micronutrient profiles across cuisines, diets, and meal types.  
- **Recommendation Systems** – Build personalized recipe recommenders based on diet, health labels, and ingredient preferences.  
- **Culinary Trend Analysis** – Track global food trends by cuisine type, dish type, and health-conscious classifications.  
- **Machine Learning Applications** – Train models for recipe generation, dietary classification, and ingredient substitution.  

## Dataset Structure  

The dataset is provided in **CSV and JSON formats** with the following categories of fields:  

### Recipe Information  
- `recipe_name`, `source`, `url`, `image_url`  
- `servings`, `calories`, `total_weight_g`  

### Classification Labels  
- `diet_labels`, `health_labels`, `cautions`  
- `cuisine_type`, `meal_type`, `dish_type`  

### Ingredients & Nutrition  
- `ingredient_lines` (raw text)  
- `ingredients` (structured: quantity, measure, food, weight)  
- `total_nutrients` (full nutrient breakdown per serving)  
- `daily_values` (recommended % values)  
- `digest` (detailed nutrition digest)  

## Source Data  

The dataset was curated from publicly available recipe and nutrition data. No proprietary or paywalled data was included. All data is intended for **non-commercial research and educational purposes** under fair use.  

## Dataset Card Contact  

📧 **contact@datahive.ai**  
