# Step 1: Setting Up the Python Application
import os
import json
import re
from typing import Union
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = "voc-62161097126677383455067ac76d27f0ad0.64235909"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

def read_file(filename: str) -> Union[str, None]:
    try:
        with open(f"data/{filename}", 'r') as file:
            content = file.read()
            return content if content.strip() else None
    except FileNotFoundError:
        return None


def write_file(filename: str, content: str) -> None:
    with open(f"data/{filename}", 'w') as file:
        file.write(content)


def safe_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return None

GENERATED_LISTINGS_FILENAME = "2-listings.txt"
LISTINGS_EMBEDDINGS_FILENAME = "2b-listings-embeddings.json"
PREFERENCES_EMBEDDINGS_FILENAME = "5-preferences-embeddings.json"
CUSTOM_DESCRIPTIONS_FILENAME = "6-custom-descriptions.json"

chat_model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Step 2: Generating Real Estate Listings
listings_str = read_file(GENERATED_LISTINGS_FILENAME)
if listings_str is None:
    print("02: Generating listings")
    generate_listings_message = """
    Generate 10 more real estate listings in json format following the example below:
    { 
        "Neighborhood": "Green Oaks",
        "Price": 800000,
        "Bedrooms": 3,
        "Bathrooms": 2.5
        "Square Footage": 2000,
        "Description": "Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.",
        "Neighborhood Description": "Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze."
    }
    """
    response = chat_model.invoke(generate_listings_message)
    listings_str = response.text()
    write_file(GENERATED_LISTINGS_FILENAME, listings_str)
else:
    print("02: Loaded listings from file")

listings_str_json = re.search(r'```json\n(.*?)\n```', listings_str, re.DOTALL).group(1).strip()
listings = json.loads(listings_str_json)

from langchain_openai import OpenAIEmbeddings
import lancedb

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Generate embeddings
embeddings_str = read_file(LISTINGS_EMBEDDINGS_FILENAME)
if embeddings_str is None:
    print("2b: Generating embeddings")
    embeddings = []
    for listing in listings:
        embedding = embeddings_model.embed_query(json.dumps(listing))
        embeddings.append(embedding)
    write_file(LISTINGS_EMBEDDINGS_FILENAME, json.dumps(embeddings))
else:
    print("2b: Loading embeddings from file")
    embeddings = json.loads(embeddings_str)

for i, listing in enumerate(listings):
    listing["vector"] = embeddings[i]

# Step 3: Storing Listings in a Vector Database
uri = "data/3-realestate-lancedb"
db = lancedb.connect(uri)
listings_tbl = safe_call(db.open_table, "listings")
if listings_tbl == None:
    listings_tbl = db.create_table("listings", data=listings)
    print("3: Creating table")
else:
    print("3: Table already exists")

# Step 4: Building the User Preference Interface
questions = [
    "How big do you want your house to be?"
    "What are 3 most important things for you in choosing this property?",
    "Which amenities would you like?",
    "Which transportation options are important to you?",
    "How urban do you want your neighborhood to be?",
]
answers = [
    "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
    "A quiet neighborhood, good local schools, and convenient shopping options.",
    "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
    "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
    "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
]

# Step 5: Searching Based on Preferences
preferences_embedding_str = read_file(PREFERENCES_EMBEDDINGS_FILENAME)
if preferences_embedding_str is None:
    print("5: Generating preferences embeddings")
    preferences_embedding = embeddings_model.embed_query(" ".join(answers))
    write_file(PREFERENCES_EMBEDDINGS_FILENAME, json.dumps(preferences_embedding))
else:
    print("5: Loading preferences embeddings from file")
    preferences_embedding = json.loads(preferences_embedding_str)

matched_listings = listings_tbl.search(preferences_embedding).limit(3).to_list()
for matched_listing in matched_listings:
    # Just to make the output more readable
    matched_listing.pop("vector")

# Step 6: Personalizing Listing Descriptions
matched_listings_str = read_file(CUSTOM_DESCRIPTIONS_FILENAME)
if matched_listings_str is None:
    print("6: Generating custom descriptions")
    for matched_listing in matched_listings:
        response = chat_model.invoke(
            f"""
            Rewrite the real estate listing description, which is after __LISTING__, so that it appeals to someone with the following
            buying preferences:
            
            ${" ".join(answers)}
            
            __LISTING__
            ${matched_listing["Neighborhood"]}
            ${matched_listing["Description"]}
            ${matched_listing["Neighborhood Description"]}
            """)
        matched_listing["Customized Description"] = response.text()

    matched_listings_str = json.dumps(matched_listings)
    write_file(CUSTOM_DESCRIPTIONS_FILENAME, matched_listings_str)
    # Produce the file according to the "Project Instructions"
    write_file("listings", matched_listings_str)
else:
    print("6: Loading custom descriptions from file")
    matched_listings = json.loads(matched_listings_str)

print(matched_listings_str)