def format_sources(docs):
    """Format retrieved documents without duplicates."""
    if not docs:
        return "No source documents found."
    
    # Remove duplicate titles
    seen_titles = set()
    unique_docs = []
    
    for d in docs:
        title = d.metadata.get("title", "Unknown")
        if title not in seen_titles:
            seen_titles.add(title)
            unique_docs.append(d)
    
    if not unique_docs:
        return "No unique sources found."
    
    md = ""
    for i, d in enumerate(unique_docs, 1):
        title = d.metadata.get("title", "Unknown")
        snippet = d.page_content[:200].strip().replace("\n", " ")
        md += f"{i}. **{title}**\n   _{snippet}..._\n\n"
    
    return md.strip()


# Test function
if __name__ == "__main__":
    print("✅ utils.py loaded successfully")