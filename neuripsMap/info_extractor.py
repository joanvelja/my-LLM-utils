import re
from typing import List

import pandas as pd


def extract_session_info(text: str, paper_start_pos: int) -> dict | None:
    # Pattern to match session details above paper
    session_pattern = r"### Poster Session\n\nPoster Session (\d+)([A-Za-z]+)\n---------------------\n\n##### ([^\n]+)\n\n([^\n]+)"

    # Find all session headers
    sessions = list(re.finditer(session_pattern, text))

    # Find the closest session header before paper
    closest_session = None
    for session in sessions:
        if session.end() < paper_start_pos:
            closest_session = session
        else:
            break

    if closest_session:
        return {
            "session_number": closest_session.group(1),
            "session_wing": closest_session.group(2),
            "location": closest_session.group(3),
            "time": closest_session.group(4),
        }
    return None


def extract_papers(
    text: str, authors_of_interest: List[str], titles_crossref: List[str] = []
) -> pd.DataFrame:
    paper_pattern = (
        r"#{5}\s+\*\*\[([^\]]+)\]\(([^\)]+)\)\*\*\n\n(.*?)\n\n(.*?)(?=\n#{5}|\Z)"
    )

    papers = []

    for match in re.finditer(paper_pattern, text, re.DOTALL):
        title = match.group(1)
        url = f"www.neurips.cc{match.group(2)}"
        authors = match.group(3)
        abstract = match.group(4)

        # Postprocess abstract
        abstract = abstract.split("\n\n  \n\n")[0]

        # Get session info for this paper
        session_info = extract_session_info(text, match.start())

        # Extract all papers if no filtering criteria provided
        if (not authors_of_interest and not titles_crossref) or (
            any(author.lower() in authors.lower() for author in authors_of_interest)
            or any(
                title.lower() in title_crossref.lower()
                for title_crossref in titles_crossref
                if titles_crossref is not None
            )
        ):
            paper_data = {
                "title": title,
                "url": url,
                "authors": authors,
                "abstract": abstract,
            }

            # Add session info if found
            if session_info:
                paper_data.update(
                    {
                        "session_number": session_info["session_number"],
                        "session_wing": session_info["session_wing"],
                        "location": session_info["location"],
                        "time": session_info["time"],
                    }
                )

            papers.append(paper_data)

    # Create DataFrame
    df = pd.DataFrame(papers)
    return df
