import requests
from bs4 import BeautifulSoup
import pandas as pd

# List of URLs with foreign speeches referencing Hindu philosophy
speech_urls = [
    "https://www.josephcampbell.org/archives/speeches",  # Joseph Campbell
    "https://www.eckharttolle.com/about/eckhart-tolle/quotes/",  # Eckhart Tolle
    "https://www.brainpickings.org/2014/11/25/einstein-jerusalem-speech/",  # Albert Einstein
    "https://www.meditationhall.com/zenmaster-thich-nhat-hanh-speech",  # Thich Nhat Hanh
    "https://www.deepakchopra.com/articles/"  # Deepak Chopra
]

# Function to scrape speeches from a given URL
def scrape_speech(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")  # Extract paragraphs
    extracted_speeches = [para.get_text(strip=True) for para in paragraphs]

    return extracted_speeches

# Collect speeches from multiple URLs
speech_data = []
for url in speech_urls:
    speeches = scrape_speech(url)
    for speech in speeches:
        speech_data.append({
            "Speaker": url.split("/")[2],  # Extract domain as speaker name
            "Source": url,
            "Speech_Excerpt": speech
        })

# Convert collected speeches into a DataFrame
df = pd.DataFrame(speech_data)

# Save dataset to CSV file
csv_file_path = "foreign_speeches.csv"
df.to_csv(csv_file_path, index=False, encoding="utf-8")

print(f"Dataset saved as {csv_file_path}")