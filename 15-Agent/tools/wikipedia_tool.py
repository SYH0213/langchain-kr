import wikipediaapi
from langchain.tools import BaseTool
from typing import Optional


class WikipediaTool(BaseTool):
    name: str = "search_wikipedia"
    description: str = """
    Search Wikipedia for information about a topic.
    Input should be a search query (e.g., 'Eiffel Tower', 'Korean cuisine', 'Tokyo history').
    Returns a summary of the Wikipedia article if found.
    """
    
    def _run(self, query: str) -> str:
        """Search Wikipedia for information."""
        try:
            wiki = wikipediaapi.Wikipedia(
                language='en',
                user_agent='LangChainTravelAgent/1.0'
            )
            
            page = wiki.page(query)
            
            if not page.exists():
                # Try to search for similar pages
                search_query = query.lower().replace(" ", "_")
                page = wiki.page(search_query)
                
                if not page.exists():
                    return f"No Wikipedia article found for '{query}'. Try rephrasing your search or being more specific."
            
            # Get summary (first 500 characters for brevity)
            summary = page.summary
            if len(summary) > 1000:
                summary = summary[:1000] + "..."
            
            # Get categories for context
            categories = list(page.categories.keys())[:5]
            categories_str = ", ".join([cat.replace("Category:", "") for cat in categories]) if categories else "N/A"
            
            result = f"""
Wikipedia: {page.title}
URL: {page.fullurl}

Summary:
{summary}

Categories: {categories_str}

For more detailed information, you can visit the full article at: {page.fullurl}
"""
            return result
            
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of Wikipedia tool (not implemented)."""
        return self._run(query)