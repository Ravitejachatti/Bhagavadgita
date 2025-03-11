import requests
from bs4 import BeautifulSoup
import pandas as pd

# List of URLs containing ISKCON speech transcripts
speech_urls = [
    "https://prabhupadavani.org/transcriptions/",  # Srila Prabhupada lectures
    "https://vedabase.io/en/library/transcripts/",  # Vedabase transcripts
    "https://www.radhanathswami.net/lectures/",  # Radhanath Swami lectures
]

# Function to scrape speech text from a webpage
def scrape_speech(url):
    response = requests.get(url)
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")  # Extract all paragraphs
    extracted_speeches = [para.get_text().strip() for para in paragraphs if para.get_text().strip()]
    
    return extracted_speeches

# Collect speeches from multiple URLs
speech_data = []
for url in speech_urls:
    speeches = scrape_speech(url)
    for speech in speeches:
        speech_data.append({
            "Speaker": "Prabhupada" if "prabhupadavani" in url else "ISKCON Devotee",
            "Source": url,
            "Speech_Excerpt": speech
        })

# Convert collected speeches into a DataFrame
df = pd.DataFrame(speech_data)

# Save dataset to CSV file
dataset_file_path = "iskcon_speeches.csv"
df.to_csv(dataset_file_path, index=False, encoding="utf-8")

print(f"Dataset saved as {dataset_file_path}")