import random
import pandas as pd

# Define templates for generating feedback
POSITIVE_TEMPLATES = [
    "The {product_type} is absolutely fantastic! It {positive_verb} my expectations and the {positive_feature} is top-notch. I highly recommend it.",
    "I'm so impressed with the {product_type}. The {positive_feature} is excellent and it was delivered {positive_adverb}. A truly great buy!",
    "This {product_type} works perfectly. The quality is amazing and the customer service was very helpful. Five stars!",
    "Excellent {product_type}! The {positive_feature} is just what I needed. It's {positive_adj} and easy to use.",
    "I love this {product_type}! It has made my life so much easier. The {positive_feature} is a game-changer."
]

NEGATIVE_TEMPLATES = [
    "I am very disappointed with the {product_type}. It {negative_verb} after just a few uses and the {negative_feature} is terrible. I want a refund.",
    "This {product_type} is a waste of money. The {negative_feature} is poor and it {negative_verb} working quickly. Do not buy this product.",
    "Terrible {product_type}. It arrived {negative_adverb} and was damaged. The customer support was unhelpful.",
    "The {product_type} did not meet my expectations at all. The {negative_feature} is flimsy and it feels cheaply made. I regret this purchase.",
    "Worst {product_type} ever. It {negative_verb} constantly and the {negative_feature} is non-existent. Avoid at all costs."
]

NEUTRAL_TEMPLATES = [
    "The {product_type} is okay. It does what it's supposed to do, but the {neutral_feature} is nothing special. It's an average product.",
    "I have mixed feelings about the {product_type}. While the {neutral_feature} is decent, it could be better. It's just alright.",
    "This {product_type} is adequate. It serves its purpose but doesn't particularly stand out. The price was reasonable for what it is.",
    "The {product_type} is as described. The {neutral_feature} is standard and it functions as expected. No complaints, but no praises either.",
    "It's a functional {product_type}. The {neutral_feature} is acceptable for the price. I'm neither thrilled nor disappointed."
]

# Define placeholder replacements
PRODUCT_TYPES = ["item", "product", "device", "gadget", "service", "appliance"]
POSITIVE_VERBS = ["exceeded", "surpassed", "outperformed"]
POSITIVE_FEATURES = ["design", "performance", "durability", "user-friendliness", "battery life", "speed", "efficiency"]
POSITIVE_ADVERBS = ["quickly", "promptly", "efficiently"]
POSITIVE_ADJ = ["reliable", "sturdy", "effective"]

NEGATIVE_VERBS = ["broke", "stopped", "failed", "malfunctioned"]
NEGATIVE_FEATURES = ["quality", "build", "material", "functionality", "support", "instructions"]
NEGATIVE_ADVERBS = ["late", "slowly"]

NEUTRAL_FEATURES = ["build quality", "feature set", "overall design", "performance"]


def generate_feedback(template_list, sentiment_label):
    template = random.choice(template_list)
    product_type = random.choice(PRODUCT_TYPES)
    
    text = template.format(
        product_type=product_type,
        positive_verb=random.choice(POSITIVE_VERBS) if "{positive_verb}" in template else "",
        positive_feature=random.choice(POSITIVE_FEATURES) if "{positive_feature}" in template else "",
        positive_adverb=random.choice(POSITIVE_ADVERBS) if "{positive_adverb}" in template else "",
        positive_adj=random.choice(POSITIVE_ADJ) if "{positive_adj}" in template else "",
        negative_verb=random.choice(NEGATIVE_VERBS) if "{negative_verb}" in template else "",
        negative_feature=random.choice(NEGATIVE_FEATURES) if "{negative_feature}" in template else "",
        negative_adverb=random.choice(NEGATIVE_ADVERBS) if "{negative_adverb}" in template else "",
        neutral_feature=random.choice(NEUTRAL_FEATURES) if "{neutral_feature}" in template else ""
    )
    return {"text": text, "sentiment": sentiment_label}

def generate_dataset(num_samples=1000):
    data = []
    samples_per_category = num_samples // 3
    
    for _ in range(samples_per_category):
        data.append(generate_feedback(POSITIVE_TEMPLATES, "Positive"))
    
    for _ in range(samples_per_category):
        data.append(generate_feedback(NEGATIVE_TEMPLATES, "Negative"))
        
    # Adjust for any rounding issues to ensure exactly num_samples
    remaining_samples = num_samples - (2 * samples_per_category)
    for _ in range(remaining_samples):
        data.append(generate_feedback(NEUTRAL_TEMPLATES, "Neutral"))
        
    random.shuffle(data)
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate the dataset
    feedback_df = generate_dataset(num_samples=1000)
    
    # Save the dataset to a CSV file (optional, can also be used in-memory)
    # For this assignment, we'll keep it in memory for the main script
    # but it's good practice to know how to save it.
    # feedback_df.to_csv("../data/synthetic_customer_feedback.csv", index=False)
    
    print(f"Generated {len(feedback_df)} feedback entries.")
    print(feedback_df["sentiment"].value_counts())
    print("\nFirst 5 entries:")
    print(feedback_df.head())

