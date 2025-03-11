import pandas as pd
import random
import nltk
from nltk.corpus import wordnet

# Download NLTK WordNet if not already downloaded
nltk.download('wordnet')

# ✅ Function 1: Synonym Replacement
def synonym_replacement(text):
    """
    Replaces words in the given text with their synonyms from WordNet.
    """
    words = text.split()
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            new_word = synonyms[0].lemmas()[0].name()  # Pick first synonym
            new_words.append(new_word)
        else:
            new_words.append(word)
    return " ".join(new_words)

# ✅ Function 2: Random Insertion
def random_insertion(text):
    """
    Inserts a random synonym of a word in the text.
    """
    words = text.split()
    if len(words) < 2:
        return text  # No modification if too short

    index = random.randint(0, len(words) - 1)
    synonyms = wordnet.synsets(words[index])
    
    if synonyms:
        new_word = synonyms[0].lemmas()[0].name()
        words.insert(index, new_word)  # Insert synonym at random position
    
    return " ".join(words)

# ✅ Function 3: Random Swap
def random_swap(text):
    """
    Swaps two random words in the text.
    """
    words = text.split()
    if len(words) < 2:
        return text  # No modification if too short

    idx1, idx2 = random.sample(range(len(words)), 2)
    words[idx1], words[idx2] = words[idx2], words[idx1]  # Swap words
    return " ".join(words)

# ✅ Function 4: Random Deletion
def random_deletion(text):
    """
    Randomly deletes a word from the text with a probability.
    """
    words = text.split()
    if len(words) < 2:
        return text  # No modification if too short

    if random.uniform(0, 1) > 0.3:  # 30% chance to delete
        del words[random.randint(0, len(words) - 1)]

    return " ".join(words)

# ✅ Test Augmentation Functions
sample_text = "Life is about duty and responsibility."
print("Original:", sample_text)
print("Synonym Replacement:", synonym_replacement(sample_text))
print("Random Insertion:", random_insertion(sample_text))
print("Random Swap:", random_swap(sample_text))
print("Random Deletion:", random_deletion(sample_text))

# ==============================================================
#  📌 Load Bhagavad Gita Dataset
# ==============================================================
df = pd.read_csv("corrected_dataset.csv")

# ✅ Function to randomly apply one augmentation
def augment_text(text):
    augmentation_functions = [
        synonym_replacement,
        random_insertion,
        random_swap,
        random_deletion
    ]
    chosen_func = random.choice(augmentation_functions)
    return chosen_func(text)

# ✅ Generate augmented versions of the Speech column
df["Augmented_Speech"] = df["Speech"].apply(augment_text)

# ✅ Duplicate dataset by appending augmented data
df_augmented = df.copy()
df_augmented["Speech"] = df_augmented["Augmented_Speech"]

# ✅ Combine original and augmented dataset
df_combined = pd.concat([df, df_augmented], ignore_index=True)

# ✅ Drop unnecessary column
df_combined.drop(columns=["Augmented_Speech"], inplace=True)

# ✅ Save the new dataset
df_combined.to_csv("corrected_dataset_augmented.csv", index=False)

print("✅ Data augmentation complete. New dataset saved as 'corrected_dataset_augmented.csv'")