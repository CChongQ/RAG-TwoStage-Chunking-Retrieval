"""Structure-based chunking for Level 1 RAG retrieval."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, TypedDict


class Section(TypedDict):
    title: str
    content: str


class SectionRecord(TypedDict):
    document_id: str
    section: Section


@dataclass
class StructureBasedChunker:
    """Extract coherent document sections from HTML or pseudo-HTML text."""

    skip_table_classes: tuple[str, ...] = ("navbox", "sidebar", "infobox")

    def _soup(self, text: str):
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError("StructureBasedChunker requires beautifulsoup4.") from exc

        return BeautifulSoup(str(text), "html.parser")

    def _process_table(self, table: Any) -> str:
        """Convert a table into compact prose, skipping navigation tables."""

        try:
            # Skip navigation tables
            table_classes = table.get("class", []) or []
            if any(table_class in self.skip_table_classes for table_class in table_classes):
                return ""

            # 1. Extract caption
            caption = table.find("caption")
            caption_text = caption.get_text(" ", strip=True) if caption else "the data"

            # 2. Extract headers (th or first row's td)
            header_row = table.find("tr")
            headers = []
            if header_row:
                headers = [cell.get_text(" ", strip=True) for cell in header_row.find_all(["th", "td"])]

            # 3. Process all rows
            rows = []
            for row_index, row in enumerate(table.find_all("tr")[1 if headers else 0 :], start=1):
                cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
                if not cells or not any(cell.strip() for cell in cells):
                    continue

                # Build sentence for each row
                if headers:
                    # With headers: "ColumnA is value1, ColumnB is value2"
                    row_parts = []
                    for cell_index, value in enumerate(cells):
                        if not value.strip():
                            continue
                        header = headers[cell_index] if cell_index < len(headers) else f"Column {cell_index + 1}"
                        row_parts.append(f"{header} is {value}")
                    if row_parts:
                        rows.append(", ".join(row_parts) + ".")
                else:
                    # Without headers: "Row 1: value1, value2..."
                    rows.append(f"Row {row_index}: {', '.join(cells)}.")

            # 4. Combine into final paragraph
            if not rows:
                return ""
            return f"Table '{caption_text}' shows: " + " ".join(rows)
        except Exception as exc:  # pragma: no cover - defensive parity with notebook
            logging.error("Table processing error: %s", exc)
            return ""

    def clean_text(self, text: str) -> str:
        """Remove tags, citation markers, and extra whitespace from text."""
        soup = self._soup(text)
        cleaned = soup.get_text(" ", strip=True)
        # Additionally clean [1][2] text references
        cleaned = re.sub(r"\[\d+\]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _remove_reference_elements(self, soup: Any) -> None:
        # Enhanced reference removal - handle all known reference forms
        reference_selectors = [
            # Wikipedia standard references
            "sup.reference",
            "span.mw-cite-backlink",
            # General citation markers
            "span.citation",
            "span.footnote",
            "div.footnotes",
            # Reference blocks
            "ol.references",
            "div.reflist",
            "div.refbegin",
            # Hidden content
            "div.noprint",
            "span.mw-editsection",
            # Citation links
            'a[href^="#cite"]',
            'a[href*="wikisource"]',
            # Modern HTML5 notes
            '[role="doc-noteref"]',
            '[role="doc-endnotes"]',
        ]
        for selector in reference_selectors:
            for element in soup.select(selector):
                element.decompose()

    def _element_text(self, element: Any) -> str:
        if getattr(element, "name", None) == "table":
            return self._process_table(element)
        return element.get_text(" ", strip=True) if hasattr(element, "get_text") else str(element).strip()

    def _section_title(self, hierarchy: list[str], current_h4: str | None = None) -> str:
        parts = [part for part in hierarchy if part]
        if current_h4:
            parts.append(current_h4)
        return " > ".join(parts) if parts else "Full Document"

    def extract_sections(self, text: str) -> list[SectionRecord]:
        """Extract structure-based section records from a document string."""
        soup = self._soup(text)
        self._remove_reference_elements(soup)

        sections: list[SectionRecord] = []
        hierarchy: list[str] = []
        current_h4: str | None = None
        current_content: list[str] = []

        def flush() -> None:
            # Save current section
            nonlocal current_content
            content = self.clean_text(" ".join(current_content))
            if hierarchy and content:
                sections.append(
                    {
                        "document_id": str(uuid.uuid4()),
                        "section": {
                            "title": self._section_title(hierarchy, current_h4),
                            "content": content,
                        },
                    }
                )
            current_content = []

        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol", "table"]):
            name = element.name.lower()
            if name in {"h1", "h2", "h3"}:
                flush()
                # Update hierarchy
                level = int(name[1])
                title = element.get_text(" ", strip=True)
                hierarchy = hierarchy[: level - 1] + [title]
                current_h4 = None
            elif name == "h4":
                # Set new H4 as current
                flush()
                current_h4 = element.get_text(" ", strip=True)
            else:
                content = self._element_text(element)
                if content:
                    current_content.append(content)

        flush()

        if not sections:
            # Fallback: if no sections were found, return the entire document
            full_text = self.clean_text(str(soup))
            if full_text:
                sections.append(
                    {
                        "document_id": str(uuid.uuid4()),
                        "section": {"title": "Full Document", "content": full_text},
                    }
                )

        return sections

    def chunk_documents(
        self,
        documents: list[dict[str, Any]],
        *,
        max_documents: int | None = None,
    ) -> list[SectionRecord]:
        """Extract sections from a list of dataset records."""
        selected_documents = documents[:max_documents] if max_documents is not None else documents
        all_sections: list[SectionRecord] = []

        for doc_index, document in enumerate(selected_documents):
            text = next(
                (
                    document[field]
                    for field in ("document_text", "html_content", "text", "content")
                    if document.get(field)
                ),
                str(document),
            )
            sections = self.extract_sections(text)
            for section_index, section in enumerate(sections, start=1):
                all_sections.append(
                    {
                        "document_id": f"Doc_{doc_index}_Sec_{section_index}",
                        "section": section["section"],
                    }
                )

        return all_sections

    def to_langchain_documents(self, sections: list[SectionRecord]) -> list[Any]:
        """Convert section records to LangChain ``Document`` objects."""
        try:
            from langchain_core.documents import Document
        except ImportError:
            try:
                from langchain.schema import Document
            except ImportError as exc:
                raise ImportError("to_langchain_documents requires langchain-core or langchain.") from exc

        # Convert sections to LangChain documents
        return [
            Document(
                page_content=section["section"]["content"],
                metadata={
                    "title": section["section"]["title"],
                    "source": section["document_id"],
                },
            )
            for section in sections
        ]
