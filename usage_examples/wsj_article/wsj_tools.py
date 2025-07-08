"""
Tools for WSJ article finder workflow.

These tools provide search functionality and article fetching capabilities.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from agent_workflow.providers import tool

logger = logging.getLogger(__name__)


@tool(name="search_web", description="Search the web for information")
def search_web(query: str) -> Dict[str, Any]:
    # Simulate search results for WSJ articles
    # Extract potential keywords from the query
    keywords = query.lower().split()
    wsj_related = any(
        kw in query.lower() for kw in ["wsj", "wall street journal", "journal"]
    )

    results = []

    # Generate simulated search results based on query
    if wsj_related:
        # Generate WSJ-specific results
        results = [
            {
                "title": "The Future of AI in Finance - WSJ Digital Edition",
                "url": "https://www.wsj.com/articles/future-ai-finance-markets-investing-234567890",
                "snippet": "Wall Street firms are rapidly adopting artificial intelligence to transform how they analyze markets, pick stocks and serve clients.",
                "published_date": "2023-10-15",
                "source": "Wall Street Journal",
            },
            {
                "title": "Tech Giants Race to Dominate AI Market",
                "url": "https://www.wsj.com/tech/ai-competition-big-tech-firms-123456789",
                "snippet": "Google, Microsoft and others are investing billions in a high-stakes battle to lead the next computing revolution.",
                "published_date": "2023-09-28",
                "source": "Wall Street Journal",
            },
            {
                "title": "How Climate Policy Is Reshaping Energy Markets",
                "url": "https://www.wsj.com/business/energy-climate-policy-markets-345678901",
                "snippet": "New regulations and incentives are accelerating the transition to renewable energy sources and challenging traditional business models.",
                "published_date": "2023-10-05",
                "source": "Wall Street Journal",
            },
            {
                "title": "Global Supply Chain Disruptions Continue to Challenge Retailers",
                "url": "https://www.wsj.com/business/retail-supply-chain-challenges-456789012",
                "snippet": "Major retailers are still struggling with inventory management as global logistics networks remain strained.",
                "published_date": "2023-10-12",
                "source": "Wall Street Journal",
            },
            {
                "title": "Healthcare Spending Reaches New Highs Amid Industry Transformation",
                "url": "https://www.wsj.com/health/healthcare-spending-trends-567890123",
                "snippet": "Hospital systems and insurers are adapting to post-pandemic realities with new business models and technology investments.",
                "published_date": "2023-09-20",
                "source": "Wall Street Journal",
            },
        ]
    else:
        # Generic news results
        results = [
            {
                "title": "Latest Economic Indicators Point to Slowing Growth",
                "url": "https://www.newssite.com/economy/indicators-slow-growth",
                "snippet": "Key economic metrics suggest the pace of expansion is moderating across major sectors.",
                "published_date": "2023-10-14",
                "source": "Financial News Today",
            },
            {
                "title": "Federal Reserve Signals Potential Rate Changes",
                "url": "https://www.economistnews.com/fed-policy-outlook",
                "snippet": "Central bank officials discuss monetary policy adjustments in response to recent economic data.",
                "published_date": "2023-10-10",
                "source": "Economist News",
            },
        ]

    # Add one result that matches keywords if they exist
    if len(keywords) >= 2:
        keyword1 = keywords[0]
        keyword2 = keywords[1] if len(keywords) > 1 else keywords[0]

        custom_result = {
            "title": f"{keyword1.capitalize()} Trends Show Impact on {keyword2.capitalize()} Markets",
            "url": f"https://www.wsj.com/articles/{keyword1}-{keyword2}-impacts-markets-987654321",
            "snippet": f"Recent developments in {keyword1} are creating significant ripple effects across {keyword2} sectors, experts say.",
            "published_date": datetime.now().strftime("%Y-%m-%d"),
            "source": "Wall Street Journal",
        }

        # Insert at front of results
        results.insert(0, custom_result)

    return {"query": query, "results_count": len(results), "results": results}


@tool(name="fetch_article", description="Fetch the content of a news article")
async def fetch_article(url: str) -> Dict[str, Any]:
    """
    Simulate fetching and parsing a news article.

    Args:
        url: URL of the article to fetch

    Returns:
        Parsed article content
    """
    logger.info(f"Fetching article: {url}")

    # In a real implementation, this would fetch and parse the actual article
    # For demo purposes, we'll simulate content based on the URL

    # Extract potential topics from the URL
    url_parts = url.lower().split("/")
    url_parts = [
        part
        for part in url_parts
        if part and part not in ("www", "com", "org", "https:", "http:")
    ]

    # Generate article parameters based on URL components
    keywords = []
    # topics = []

    for part in url_parts:
        if "wsj" in part:
            continue
        if "-" in part:
            keywords.extend(part.split("-"))
        else:
            keywords.append(part)

    # Clean up keywords
    keywords = [k for k in keywords if len(k) > 3 and not k.isdigit()][:5]

    # Generate simulated article based on keywords
    if "articles" in url_parts and len(keywords) >= 2:
        # Generate WSJ-style article
        title = " ".join([k.capitalize() for k in keywords[:3]])
        if not title.startswith("The ") and len(title.split()) < 7:
            title = "The " + title

        paragraphs = []

        # Generate intro paragraph
        paragraphs.append(
            f"By combining {keywords[0]} with innovative approaches to {keywords[1]}, "
            f"companies are finding new opportunities in today's challenging market environment. "
            f"Investors are taking notice, with several key players emerging as potential winners in this transformation."
        )

        # Add more paragraphs based on keywords
        if len(keywords) >= 3:
            paragraphs.append(
                f'"The intersection of {keywords[0]} and {keywords[2]} represents one of the most '
                f"significant opportunities we've seen in years,\" said Jane Smith, chief strategist at Capital Investments. "
                f'"Those who move quickly to adapt will likely capture significant market share."'
            )

        paragraphs.append(
            f"Data shows that investment in {keywords[0]}-related technologies has increased by 37% "
            f"year-over-year, reaching $12.5 billion in the last quarter alone. Much of this growth "
            f"has been concentrated in sectors that traditionally haven't been associated with cutting-edge "
            f"innovation."
        )

        paragraphs.append(
            f"The trend isn't limited to U.S. markets. Companies across Europe and Asia are also "
            f"exploring how {keywords[1]} can drive efficiency and create new revenue streams. "
            f"Regulatory frameworks, however, remain inconsistent across regions, creating both "
            f"challenges and opportunities for multinational organizations."
        )

        if len(keywords) >= 4:
            paragraphs.append(
                f"Critics point out that the rapid pace of change may create risks that aren't immediately "
                f'apparent. "We\'re still learning about the long-term implications of {keywords[3]}," '
                f'noted regulatory expert Robert Johnson at a recent industry conference. "Proper governance '
                f'and oversight will be essential."'
            )

        # Conclusion
        paragraphs.append(
            "As markets continue to evolve, the companies that successfully navigate these changes "
            "will likely emerge stronger. The next six months could prove pivotal as new technologies "
            "mature and adoption accelerates across industries."
        )

        # Create author name from URL components
        authors = ["Michael Johnson", "Sarah Zhang", "David Williams", "Jennifer Patel"]

        # Assemble the article
        article = {
            "url": url,
            "title": title,
            "authors": authors[hash(url) % len(authors)],
            "published_date": "2023-10-15",
            "content": "\n\n".join(paragraphs),
            "word_count": sum(len(p.split()) for p in paragraphs),
            "topics": [k.capitalize() for k in keywords[:3]],
            "publication": "The Wall Street Journal",
        }

        return article
    else:
        # Generic article if URL doesn't contain enough information
        return {
            "url": url,
            "title": "Markets Respond to Economic Data",
            "authors": "WSJ Staff",
            "published_date": datetime.now().strftime("%Y-%m-%d"),
            "content": "Article content unavailable or could not be retrieved.",
            "word_count": 0,
            "topics": ["Markets", "Economy"],
            "publication": "The Wall Street Journal",
        }
