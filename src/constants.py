"""Domain constants: pantry staples, alias dictionary, cuisine/flavor keywords."""


# this is a set of ingredients that are treated as "basic pantry items", like salt, oil, flour, 
# sugar, etc. The recommender usually does not penalize a user much for missing these, because
# we just assume people already have them since they are so common
PANTRY_STAPLES = {
    "salt", "kosher salt", "pepper", "black pepper", "white pepper", "water", "ice",
    "oil", "olive oil", "vegetable oil", "canola oil", "sunflower oil",
    "butter", "sugar", "brown sugar", "white sugar",
    "flour", "all-purpose flour", "all purpose flour",
    "baking soda", "baking powder", "vanilla extract", "vanilla",
    "garlic powder", "onion powder", "cornstarch", "corn starch",
}

# bidirectional alias dictionary: maps surface forms to canonical food names
# keys are what a user might type; values are what appears in the vocabulary
# this is used to normalize the user's ingredient list
ALIASES = {
    # ranslations (CN -> EN subset for demo)
    "洋葱": "onion", "大蒜": "garlic", "蒜": "garlic", "姜": "ginger", "葱": "green onion", "生姜": "ginger",
    "番茄": "tomato", "西红柿": "tomato", "土豆": "potato", "马铃薯": "potato", "红薯": "sweet potato", "地瓜": "sweet potato",
    "鸡肉": "chicken", "鸡": "chicken", "牛肉": "beef", "猪肉": "pork", "羊肉": "lamb", "鸭肉": "duck", "火鸡": "turkey",
    "鱼": "fish", "虾": "shrimp", "三文鱼": "salmon", "鳕鱼": "cod", "金枪鱼": "tuna", "螃蟹": "crab", "贝类": "shellfish",
    "鸡蛋": "egg", "蛋": "egg", "牛奶": "milk", "黄油": "butter", "芝士": "cheese", "奶酪": "cheese", "奶油": "cream",
    "米饭": "rice", "大米": "rice", "糙米": "brown rice", "面条": "noodle", "意面": "pasta", "面粉": "flour", "面包": "bread",
    "胡萝卜": "carrot", "青椒": "bell pepper", "辣椒": "chili pepper", "菠菜": "spinach", "西兰花": "broccoli", "芹菜": "celery",
    "豆腐": "tofu", "蘑菇": "mushroom", "香菇": "shiitake mushroom", "白菜": "cabbage", "包菜": "cabbage", "生菜": "lettuce",
    "玉米": "corn", "豌豆": "pea", "黄瓜": "cucumber", "茄子": "eggplant", "南瓜": "pumpkin", "秋葵": "okra", "竹笋": "bamboo shoot",
    "苹果": "apple", "香蕉": "banana", "柠檬": "lemon", "青柠": "lime", "橙子": "orange", "草莓": "strawberry", "葡萄": "grape",
    "酱油": "soy sauce", "生抽": "soy sauce", "老抽": "dark soy sauce", "醋": "vinegar", "黑醋": "black vinegar", 
    "糖": "sugar", "红糖": "brown sugar", "盐": "salt", "蜂蜜": "honey", "黑胡椒": "black pepper", "白胡椒": "white pepper",
    "橄榄油": "olive oil", "芝麻油": "sesame oil", "香油": "sesame oil", "花生油": "peanut oil", "孜然": "cumin", "八角": "star anise",
    "西瓜": "watermelon", "哈密瓜": "melon", "芥末": "wasabi", "芥末酱": "mustard", 
    # english morphology + common variants (extended set below uses lemmatization)
    "tomatoes": "tomato", "potatoes": "potato", "onions": "onion",
    "eggs": "egg", "carrots": "carrot", "mushrooms": "mushroom",
    "chilies": "chili pepper", "chilli": "chili pepper", "chile": "chili pepper",
    "scallion": "green onion", "scallions": "green onion", "spring onion": "green onion",
    "coriander": "cilantro", "capsicum": "bell pepper", "aubergine": "eggplant",
    "courgette": "zucchini", "rocket": "arugula", "prawn": "shrimp", "prawns": "shrimp",
    # superordinate / abstract terms (expanded during matching)
    "poultry": "chicken", "seafood": "fish", "red meat": "beef",
    "greens": "spinach", "dairy": "milk",
}

# superordinate terms that expand to *sets* of canonical ingredients.
# used by the normalizer when the user types a category rather than a specific food.
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

# candidate topic labels for semantic cluster naming. Each cluster gets the
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


# flavor keywords for clustering, helping system to associate ingredients with flavor profiles
FLAVOR_KEYWORDS = {
    "spicy": ["chili", "chile", "jalapeno", "cayenne", "sriracha", "chipotle", "pepper flakes"],
    "sweet": ["sugar", "honey", "maple", "caramel"],
    "sour": ["lemon", "lime", "vinegar", "tamarind"],
    "umami": ["soy sauce", "miso", "fish sauce", "parmesan", "mushroom"],
    "smoky": ["smoked paprika", "chipotle", "bacon", "liquid smoke"],
}
