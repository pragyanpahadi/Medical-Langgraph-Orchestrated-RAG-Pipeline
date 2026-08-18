"""Turns retrieved documents into numbered [n] citations for the LLM to use
in place of raw filenames, plus a matching Sources list (real paper title,
clickable DOI link) for the clinician to check.

Numbering is per-response, not a fixed bibliography: each call assigns 1..N
to the unique sources among the documents actually retrieved for that one
query/report, in order of first appearance.
"""


def doi_url(doi: str | None) -> str | None:
    return f"https://doi.org/{doi}" if doi else None


def build_citation_map(docs: list) -> dict:
    """Maps source filename -> citation number, in order of first appearance."""
    mapping = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source not in mapping:
            mapping[source] = len(mapping) + 1
    return mapping


def format_context_with_numbers(docs: list, citation_map: dict) -> str:
    """Builds the LLM-facing context string, labeling each chunk with its
    citation number instead of its filename."""
    return "\n\n".join(
        f"[{citation_map[doc.metadata.get('source', 'unknown')]}] {doc.page_content}"
        for doc in docs
    )


def build_sources_list(docs: list, citation_map: dict) -> list:
    """One entry per unique source (not per chunk), in citation order."""
    seen = set()
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source in seen:
            continue
        seen.add(source)
        doi = doc.metadata.get("doi")
        sources.append({
            "number": citation_map[source],
            "title": doc.metadata.get("title", source),
            "doi": doi,
            "doi_url": doi_url(doi),
        })
    return sorted(sources, key=lambda s: s["number"])


def format_references_markdown(sources: list) -> str:
    """A '**References**' block, e.g. for appending to an LLM-generated report."""
    if not sources:
        return ""
    lines = []
    for s in sources:
        line = f"[{s['number']}] {s['title']}"
        if s["doi"]:
            line += f". DOI: [{s['doi']}]({s['doi_url']})"
        lines.append(line)
    return "**References**\n" + "\n".join(lines)
