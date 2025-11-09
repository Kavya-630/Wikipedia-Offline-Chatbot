import wikipedia
from langchain_core.documents import Document

def fetch_wikipedia_pages(topics, max_pages_per_topic=2):
    """Fetch and clean Wikipedia pages for given topics."""
    docs = []
    
    for topic in topics:
        try:
            print(f"[INFO] Searching for: {topic}")
            results = wikipedia.search(topic, results=max_pages_per_topic)
            
            for title in results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    content = page.content
                    
                    # Create document with metadata
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "title": title,
                            "url": page.url,
                            "topic": topic
                        }
                    ))
                    print(f"[INFO] ✅ Fetched: {title}")
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    print(f"[WARN] Disambiguation for {title}, trying first option")
                    try:
                        first_option = e.options[0]
                        page = wikipedia.page(first_option, auto_suggest=False)
                        docs.append(Document(
                            page_content=page.content,
                            metadata={
                                "title": first_option,
                                "url": page.url,
                                "topic": topic
                            }
                        ))
                        print(f"[INFO] ✅ Fetched: {first_option}")
                    except Exception as e2:
                        print(f"[ERROR] Failed to fetch {first_option}: {e2}")
                        
                except Exception as e:
                    print(f"[WARN] Could not fetch {title}: {e}")
                    
        except Exception as e:
            print(f"[ERROR] Failed to search for {topic}: {e}")
    
    print(f"[INFO] 📚 Total pages fetched: {len(docs)}")
    return docs


# Test function
if __name__ == "__main__":
    print("✅ wiki_loader.py loaded successfully")