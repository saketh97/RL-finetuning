import os
import json
from sec_api import QueryApi
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SEC_API_KEY")

query_api = QueryApi(api_key=API_KEY)

companies = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "BAC", "WFC",
    "GS", "KO", "PEP", "WMT", "TGT", "COST", "IBM", "ORCL", "INTC", "CSCO"
]


SAVE_DIR = "data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_filings(ticker):
    query = {
        "query": {
            "query_string": {
                "query": f"ticker:{ticker} AND formType:(\"10-K\" OR \"10-Q\")"
            }
        },
        "from": "0",
        "size": "5",
        "sort": [{"filedAt": {"order": "desc"}}]
    }
     
    filings = query_api.get_filings(query)
    
    return filings["filings"]

if __name__ == "__main__":
    for ticker in tqdm(companies):
        print(f"Downloading {ticker} filings...")
        filings = download_filings(ticker)

        out_path = os.path.join(SAVE_DIR, f"{ticker}.json")
        with open(out_path, "w") as f:
            json.dump(filings, f, indent=2)

    print("Download complete")