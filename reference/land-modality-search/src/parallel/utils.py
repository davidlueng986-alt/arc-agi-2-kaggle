import re

def extract_tag_content(text: str, tag_name: str) -> str | None:
    """Extracts content between <tag>...</tag>."""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
