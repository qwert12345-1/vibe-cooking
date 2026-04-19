"""Domain constants: pantry staples, alias dictionary, cuisine/flavor keywords."""

PANTRY_STAPLES = {
    "salt", "pepper", "black pepper", "white pepper", "water", "ice",
    "oil", "olive oil", "vegetable oil", "canola oil", "sunflower oil",
    "butter", "sugar", "brown sugar", "white sugar",
    "flour", "all-purpose flour", "all purpose flour",
    "baking soda", "baking powder", "vanilla extract", "vanilla",
    "garlic powder", "onion powder", "cornstarch", "corn starch",
}

# Bidirectional alias dictionary: maps surface forms to canonical food names.
# Keys are what a user might type; values are what appears in the vocabulary.
ALIASES = {
    # Translations (CN -> EN subset for demo)
    "洋葱": "onion", "大蒜": "garlic", "蒜": "garlic", "姜": "ginger",
    "番茄": "tomato", "西红柿": "tomato", "土豆": "potato", "马铃薯": "potato",
    "鸡肉": "chicken", "鸡": "chicken", "牛肉": "beef", "猪肉": "pork",
    "鱼": "fish", "虾": "shrimp", "鸡蛋": "egg", "蛋": "egg",
    "米饭": "rice", "大米": "rice", "面条": "noodle", "面粉": "flour",
    "胡萝卜": "carrot", "青椒": "bell pepper", "辣椒": "chili pepper",
    "蘑菇": "mushroom", "香菇": "shiitake mushroom", "豆腐": "tofu",
    "酱油": "soy sauce", "醋": "vinegar", "糖": "sugar", "盐": "salt",
    # English morphology + common variants (extended set below uses lemmatization)
    "tomatoes": "tomato", "potatoes": "potato", "onions": "onion",
    "eggs": "egg", "carrots": "carrot", "mushrooms": "mushroom",
    "chilies": "chili pepper", "chilli": "chili pepper", "chile": "chili pepper",
    "scallion": "green onion", "scallions": "green onion", "spring onion": "green onion",
    "coriander": "cilantro", "capsicum": "bell pepper", "aubergine": "eggplant",
    "courgette": "zucchini", "rocket": "arugula", "prawn": "shrimp", "prawns": "shrimp",
    # Superordinate / abstract terms (expanded during matching)
    "poultry": "chicken", "seafood": "fish", "red meat": "beef",
    "greens": "spinach", "dairy": "milk",
}

# Superordinate terms that expand to *sets* of canonical ingredients.
# Used by the normalizer when the user types a category rather than a specific food.
SUPERORDINATES = {
    "poultry": ["chicken", "turkey", "duck"],
    "seafood": ["fish", "shrimp", "salmon", "tuna", "cod", "prawn"],
    "red meat": ["beef", "pork", "lamb"],
    "meat": ["chicken", "beef", "pork", "lamb", "turkey"],
    "greens": ["spinach", "kale", "lettuce", "arugula", "chard"],
    "dairy": ["milk", "cheese", "butter", "cream", "yogurt"],
    "citrus": ["lemon", "lime", "orange", "grapefruit"],
    "berries": ["strawberry", "blueberry", "raspberry", "blackberry"],
    "nuts": ["almond", "walnut", "pecan", "cashew", "peanut"],
}

CUISINES = [
    "american", "italian", "mexican", "chinese", "japanese", "thai", "indian",
    "french", "mediterranean", "middle eastern", "korean", "vietnamese",
    "south east asian", "greek", "spanish", "british", "german", "caribbean",
    "african", "nordic",
]

DIET_TAGS = [
    "Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Low-Carb",
    "Keto-Friendly", "Paleo", "Low-Sodium", "Low-Fat", "Low-Sugar",
    "High-Protein", "Pescatarian",
]

MEAL_TYPES = ["breakfast", "lunch", "dinner", "snack", "dessert", "appetizer", "side"]

# Candidate topic labels for semantic cluster naming. Each cluster gets the
# candidate whose SBERT embedding is closest to the cluster's mean recipe
# embedding. Cover enough culinary "categories" so most clusters find a decent
# match — keep phrasing concise, they show up in legends.
CLUSTER_TOPIC_CANDIDATES = [
    "soups and broths",
    "stews and braises",
    "baked goods and pastries",
    "cakes, cookies, and desserts",
    "stir-fries",
    "curries",
    "fresh salads",
    "pasta and noodle dishes",
    "pizza and flatbreads",
    "sandwiches and wraps",
    "breakfast dishes",
    "grilled and roasted meat",
    "seafood and fish dishes",
    "rice and grain bowls",
    "dips, sauces, and spreads",
    "cocktails and drinks",
    "smoothies and juices",
    "pickles and preserves",
    "dumplings and filled pastries",
    "egg dishes and omelets",
    "vegetable side dishes",
    "tacos, burritos, and mexican",
    "ramen and asian noodle soups",
]


FLAVOR_KEYWORDS = {
    "spicy": ["chili", "chile", "jalapeno", "cayenne", "sriracha", "chipotle", "pepper flakes"],
    "sweet": ["sugar", "honey", "maple", "caramel"],
    "sour": ["lemon", "lime", "vinegar", "tamarind"],
    "umami": ["soy sauce", "miso", "fish sauce", "parmesan", "mushroom"],
    "smoky": ["smoked paprika", "chipotle", "bacon", "liquid smoke"],
}
