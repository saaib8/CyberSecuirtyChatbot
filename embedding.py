import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
import os
from pathlib import Path
from langchain.schema import Document
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

os.environ["COHERE_API_KEY"] = os.getenv('COHERE_API_KEY')

def clean_garbage_text(text):
    """Remove unwanted patterns from text."""
    # Define garbage patterns
    patterns = [
        r"Instructor\(s\):\s*Muhammad.*?PhD.*?\n?\s*\d*",
        r"Muhammad\s+Rauf\s+Butt.*?PhD.*?\n?\s*\d*",
        r"^\s*\d+\s*$"  # Standalone numbers
    ]

    # Remove patterns
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)

    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()


def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of a PDF with manual page assignment."""
    doc = fitz.open(pdf_path)
    text = ""
    total_pages = len(doc)  # Get page count before closing

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        page_text = clean_garbage_text(page_text)
        # Simply assign page numbers sequentially: 1, 2, 3, etc.
        current_page = page_num + 1
        text += f"[PAGE_{current_page}]\n{page_text}\n"

    doc.close()
    return text


def parse_handouts_with_page_tracking(text):
    """Parse handouts while tracking page numbers for content sections."""
    handouts = []
    lines = text.splitlines()

    i = 0
    current_page = 1
    found_first_handout = False  # Track if we've found the first handout

    while i < len(lines):
        line = lines[i].strip()

        # Check for page markers
        if line.startswith('[PAGE_') and line.endswith(']'):
            current_page = int(re.search(r'\[PAGE_(\d+)\]', line).group(1))
            i += 1
            continue

        # Only look for HO# patterns if we haven't found the first handout yet
        if not found_first_handout:
            # Detect HO# pattern - strict regex for actual titles
            match = re.match(r"^\s*(HO#\s*[\d.]+)\s*:?\s*(.+)", line, re.IGNORECASE)
            if match:
                found_first_handout = True  # Mark that we found the first handout

                handout_num = match.group(1).strip()
                handout_title = match.group(2).strip()

                # Limit title to 50 characters - anything after is content
                if len(handout_title) > 50:
                    handout_title = handout_title[:50].strip()

                # Handle multi-line titles (but still respect 50 char limit)
                if not handout_title or len(handout_title) < 3:
                    title_lines = [handout_title] if handout_title else []
                    temp_i = i + 1

                    # Look ahead for title continuation (max 2 lines)
                    while temp_i < len(lines) and len(title_lines) < 2:
                        next_line = lines[temp_i].strip()
                        # Stop at content indicators
                        if (next_line.startswith('[PAGE_') or
                                re.match(r"^(Phase|Dear|The|In|Overview|So far|Quick|Now)", next_line, re.IGNORECASE) or
                                len(next_line) > 80 or
                                not next_line):
                            break
                        if next_line:
                            title_lines.append(next_line)
                        temp_i += 1

                    combined_title = " ".join(title_lines).strip()
                    # Still apply 50 character limit to combined title
                    handout_title = combined_title[:50].strip() if len(combined_title) > 50 else combined_title

                title = f"{handout_num}: {handout_title}" if handout_title else handout_num
                handout_start_page = current_page

                # Collect content with page tracking
                content_sections = []
                current_section_lines = []
                section_page = current_page

                i += 1  # Move to next line after title

                while i < len(lines):
                    next_line = lines[i].strip()

                    # Update page if we hit a page marker
                    if next_line.startswith('[PAGE_') and next_line.endswith(']'):
                        # Save current section before changing page
                        if current_section_lines:
                            content_sections.append(("\n".join(current_section_lines).strip(), section_page))
                            current_section_lines = []

                        # Update to new page
                        current_page = int(re.search(r'\[PAGE_(\d+)\]', next_line).group(1))
                        section_page = current_page
                        i += 1
                        continue

                    # Add line to current section (no need to check for HO# anymore)
                    if next_line or current_section_lines:
                        current_section_lines.append(lines[i])

                    i += 1

                # Save the last section
                if current_section_lines:
                    content_sections.append(("\n".join(current_section_lines).strip(), section_page))

                # Save handout with content sections
                if content_sections:
                    handouts.append({
                        "title": title,
                        "content_sections": content_sections,
                        "start_page": handout_start_page
                    })

                continue

        i += 1

    return handouts


def chunk_handouts_with_pages_global(handouts, chunk_size=1000, chunk_overlap=100, global_chunk_counter=0):
    """Modified version of your function with global chunk IDs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = []
    current_global_id = global_chunk_counter

    for handout in handouts:
        full_content = ""
        page_mapping = []

        for content, page_num in handout["content_sections"]:
            start_pos = len(full_content)
            full_content += content + "\n"
            end_pos = len(full_content)
            page_mapping.append((start_pos, end_pos - 1, page_num))

        chunks = splitter.split_text(full_content)
        current_pos = 0

        for i, chunk in enumerate(chunks):
            chunk_start = current_pos
            chunk_end = chunk_start + len(chunk)

            chunk_pages = set()
            for start_pos, end_pos, page_num in page_mapping:
                if not (chunk_end <= start_pos or chunk_start > end_pos):
                    chunk_pages.add(page_num)

            chunk_page = min(chunk_pages) if chunk_pages else handout["start_page"]

            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "title": handout["title"],
                        "chunk_id": current_global_id,
                        "page": chunk_page,
                        "pages_spanned": sorted(list(chunk_pages)) if len(chunk_pages) > 1 else [chunk_page],
                    },
                )
            )


            current_pos = chunk_start + len(chunk) - chunk_overlap
            current_global_id += 1

    return docs, current_global_id


def process_all_pdfs(pdf_directory, chunk_size=1000, chunk_overlap=100):
    """
    Process all PDFs in a directory one by one with global chunk IDs.

    Args:
        pdf_directory: Path to directory containing PDF files
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of all chunks from all PDFs with global chunk IDs
    """
    pdf_files = list(Path(pdf_directory).glob("*.pdf"))
    all_docs = []
    global_chunk_counter = 0

    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        try:
            # Use your existing pipeline
            raw_text = extract_text_from_pdf(str(pdf_path))
            handouts = parse_handouts_with_page_tracking(raw_text)

            # Debug info
            print(f"  → Found {len(handouts)} handouts")
            for handout in handouts:
                print(f"    - {handout['title']}")
            if len(handouts) == 0:
                # Show first 500 chars of raw text to debug
                print(f"  → Raw text preview: {raw_text[:500]}...")

            docs, global_chunk_counter = chunk_handouts_with_pages_global(
                handouts, chunk_size, chunk_overlap, global_chunk_counter
            )

            # Add source file info to metadata
            # for doc in docs:
            #     doc['metadata']['source_file'] = pdf_path.stem

            all_docs.extend(docs)
            print(f"  → Generated {len(docs)} chunks")


        except Exception as e:
            print(f"  ❌ Error processing {pdf_path.name}: {e}")

    print(f"\nTotal chunks created: {len(all_docs)}")
    return all_docs



all_chunks = process_all_pdfs("data")

embeddings = CohereEmbeddings(model="embed-english-v3.0")
vectorstore = FAISS.from_documents(all_chunks, embeddings)
vectorstore.save_local("faiss_index")
